import os
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, FluxPriorReduxPipeline, FluxPipeline
from tqdm import tqdm
import datetime
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
import warnings
warnings.filterwarnings('ignore')
import argparse
from diffusers.utils import load_image
from safetensors.torch import load_file
from faker import Faker
import torchvision
torchvision.disable_beta_transforms_warning()
from dotenv import load_dotenv

# Set PyTorch memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

fake = Faker()

# Load environment variables
load_dotenv()

# Enable memory-efficient attention for SD-based models
torch.backends.cuda.enable_mem_efficient_sdp(True)
dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "main"

# Memory optimization techniques
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class LyingSigmaSampler:
    def __init__(self, 
                 dishonesty_factor: float = -0.05, 
                 start_percent: float = 0.1, 
                 end_percent: float = 0.9):
        self.dishonesty_factor = dishonesty_factor
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, model, x, sigmas, **kwargs):
        start_percent, end_percent = self.start_percent, self.end_percent
        ms = model.inner_model.inner_model.model_sampling
        start_sigma, end_sigma = (
            round(ms.percent_to_sigma(start_percent), 4),
            round(ms.percent_to_sigma(end_percent), 4),
        )
        del ms

        def model_wrapper(x, sigma, **extra_args):
            sigma_float = float(sigma.max().detach().cpu())
            if end_sigma <= sigma_float <= start_sigma:
                sigma = sigma * (1.0 + self.dishonesty_factor)
            return model(x, sigma, **extra_args)

        for k in ("inner_model", "sigmas"):
            if hasattr(model, k):
                setattr(model_wrapper, k, getattr(model, k))

        return model_wrapper(x, sigmas, **kwargs)


class FluxBase:
    def __init__(self, acceleration=None, loras=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.acceleration = acceleration
        self.loras = loras or []
        self.dtype = torch.bfloat16  # Define dtype as an instance attribute
        # Load common components
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        self.vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision).to(self.device)

        # Determine steps based on acceleration
        if acceleration in ["alimama", "hyper"]:
            if acceleration == "alimama":
                TRANSFORMER_PATH_HYPER_ALIMAMA = os.getenv('TRANSFORMER_PATH_HYPER_ALIMAMA')
                if not TRANSFORMER_PATH_HYPER_ALIMAMA or not os.path.exists(TRANSFORMER_PATH_HYPER_ALIMAMA):
                    raise ValueError(f"TRANSFORMER_PATH_HYPER_ALIMAMA not set in .env or the file does not exist: {TRANSFORMER_PATH_HYPER_ALIMAMA}")
                transpath = TRANSFORMER_PATH_HYPER_ALIMAMA
            elif acceleration == "hyper":
                TRANSFORMER_PATH_HYPER = os.getenv('TRANSFORMER_PATH_HYPER')
                if not TRANSFORMER_PATH_HYPER or not os.path.exists(TRANSFORMER_PATH_HYPER):
                    raise ValueError(f"TRANSFORMER_PATH_HYPER not set in .env or the file does not exist: {TRANSFORMER_PATH_HYPER}")
                transpath = TRANSFORMER_PATH_HYPER
            self.desired_num_steps = 10
        else:
            # Get transformer path from environment variable
            TRANSFORMER_PATH = os.getenv('TRANSFORMER_PATH')
            if not TRANSFORMER_PATH or not os.path.exists(TRANSFORMER_PATH):
                raise ValueError(f"TRANSFORMER_PATH not set in .env or the file does not exist: {TRANSFORMER_PATH}")
            transpath = TRANSFORMER_PATH          
            self.desired_num_steps = 25        

        print(f"Loading transformer from: {transpath}")
        try:
            # Load transformer with manual memory management
            print("Loading transformer...")
            self.transformer = FluxTransformer2DModel.from_pretrained(
                bfl_repo, 
                subfolder="transformer", 
                torch_dtype=dtype, 
                revision=revision
            )
            state_dict = load_file(transpath, device=self.device)
            self.transformer.load_state_dict(state_dict, strict=False)
            self.transformer.eval()            
        except Exception as e:
            print(f"Error loading transformer: {e}")
            print("Falling back to default transformer from repository")
            self.transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)
            # Quantize the transformer
            print(datetime.datetime.now(), "Quantizing transformer")
            quantize(self.transformer, weights=qfloat8)
            freeze(self.transformer)

        # Initialize progress callbacks
        self.initialize_callback()
        
    def generate_docker_name(self):
            return f"{fake.word().lower()}_{fake.word().lower()}"        

    def initialize_callback(self):
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents", None)
            callback.step_pbar.update(1)
            return {"latents": latents} if latents is not None else {}
        self.callback = callback


class FluxRefine(FluxBase):
    def __init__(self, acceleration=None, loras=None):
        super().__init__(acceleration, loras)
        
        # Load Refine-specific components on CPU
        print("Loading text encoders...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", 
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            bfl_repo, 
            subfolder="text_encoder_2", 
            torch_dtype=self.dtype, 
            revision=revision,
            low_cpu_mem_usage=True
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            bfl_repo, 
            subfolder="tokenizer_2", 
            revision=revision
        )
        
        # Create pipeline with all components on CPU
        print("Creating pipeline...")
        self.pipe = FluxImg2ImgPipeline(
            scheduler=self.scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            vae=self.vae,
            transformer=self.transformer,
        )
        
        print("Setting up pipeline optimizations...")
        # Modified memory optimizations
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.enable_attention_slicing(1)
        
        # Direct VAE slicing configuration
        self.vae.enable_slicing()  # <-- This is the key line that changed
        self.vae.enable_tiling()   # <-- Add tiling for large images
        
        # Configure the pipeline's VAE directly
        self.pipe.vae = self.vae
        self.pipe.set_progress_bar_config(disable=True)
        
        # Apply detailer daemon
        self.pipe.scheduler.set_sigmas = LyingSigmaSampler(
            dishonesty_factor=-0.05,
            start_percent=0.1,
            end_percent=0.9
        )
        
        print("FluxRefine initialized")
    

    def process_image(self, input_path, output_path, prompt):
        try:
                
            print(f"Loading image: {input_path}")
            init_image = load_image(input_path)
            width, height = init_image.size
            
            # Check if image is too large and resize if necessary
            max_resolution = 1024  # Set a maximum resolution
            max_pixels = 1024 * 1024  # 1 megapixel limit
            current_pixels = width * height
            
            if current_pixels > max_pixels:
                print(f"Image too large ({width}x{height}), resizing...")
                # Calculate new dimensions while preserving aspect ratio
                aspect_ratio = width / height
                if width > height:
                    new_width = min(max_resolution, int(max_pixels**0.5))
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = min(max_resolution, int(max_pixels**0.5))
                    new_width = int(new_height * aspect_ratio)
                
                init_image = init_image.resize((new_width, new_height), Image.LANCZOS)
                width, height = init_image.size
                print(f"Resized to {width}x{height}")
            
            # Calculate steps
            strength = 0.30  # Refine uses strength 0.20
            num_inference_steps = self.desired_num_steps / strength
            
            # Set up progress bar
            with tqdm(total=self.desired_num_steps, desc=f"Steps for {os.path.basename(input_path)}", leave=True) as step_pbar:
                self.callback.step_pbar = step_pbar
                
                # Run Refine pipeline with explicit memory management
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        image=init_image,
                        num_inference_steps=int(num_inference_steps),
                        strength=strength,
                        guidance_scale=3.0,
                        height=height,
                        width=width,
                        num_images_per_prompt=1,
                        callback_on_step_end=self.callback
                    ).images
            
            # Save results
            if len(result) > 1:
                for idx, img in enumerate(result):
                    output_image_path = f"{output_path.rstrip('.png')}_{str(idx + 1).zfill(4)}.png"
                    img.save(output_image_path)
            else:
                result[0].save(output_path)
                
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


def upscale_to_sdxl(image_path):
    """
    Upscale image to nearest SDXL resolution (maintaining aspect ratio) if below 1 megapixel.
    Common SDXL resolutions: 1024x1024, 1024x576, 576x1024, 1152x896, 896x1152, etc.
    
    Args:
        image_path (str): Path to input image
    
    Returns:
        PIL.Image: Resized image object
    """
    # Open the image
    img = Image.open(image_path)
    
    # Get current dimensions
    width, height = img.size
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # SDXL base sizes to consider
    sdxl_sizes = [
        (1024, 1024),  # 1:1
        (1024, 576),   # 16:9
        (576, 1024),   # 9:16
        (1152, 896),   # 9:7
        (896, 1152),   # 7:9
        (1024, 768),   # 4:3
        (768, 1024),   # 3:4
        (1216, 832),   # Additional sizes
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (866, 1155),        
        (640, 1536)
    ]
    
    # Filter out sizes that are smaller than 1 megapixel
    sdxl_sizes = [(w, h) for w, h in sdxl_sizes if w * h >= 1000000]
    
    # Find the best matching SDXL resolution
    best_size = None
    min_ratio_diff = float('inf')
    
    for w, h in sdxl_sizes:
        current_ratio = w / h
        ratio_diff = abs(current_ratio - aspect_ratio)
        
        if ratio_diff < min_ratio_diff:
            min_ratio_diff = ratio_diff
            best_size = (w, h)
    
    # Resize image using LANCZOS resampling (high quality)
    resized_img = img.resize(best_size, Image.LANCZOS)   
    return resized_img

def process_directory(input_dir, output_dir, acceleration, prompt, generate_names=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    processor = FluxRefine(acceleration=acceleration)

    # Check if input_dir is a file or directory
    if os.path.isfile(input_dir):
        input_dir, filename = os.path.split(input_dir)
        png_files = [filename]
    elif os.path.isdir(input_dir):
        png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    else:
        raise ValueError("Input must be either a file or a directory.")

    # Create a list of files to process
    files_to_process = []
    print(f"Output directory: {output_dir}")
    
    if generate_names:
        # If generating names, process all files without checking existence
        files_to_process = png_files
    else:
        # Otherwise, check for existing files and skip them
        for f in png_files:
            filename = os.path.basename(f)
            output_path = os.path.join(output_dir, filename)
            print(f"Output path: {output_path}")
            if not os.path.exists(output_path):
                files_to_process.append(f)
            else:
                print(f"Skipping {f}: already exists in output directory.")

    # Debug: print the number of files that need processing
    print(f"Total files to process: {len(files_to_process)}")

    total_files_to_process = len(files_to_process)

    with tqdm(total=total_files_to_process, desc="Processing images", unit="img") as main_pbar:
        # Create the list of input files
        input_files = [os.path.join(input_dir, filename) for filename in files_to_process]

        # Process each file
        for input_path in input_files:
            fname = os.path.basename(input_path)
            
            # Determine output path
            if generate_names:
                # Use generate_docker_name() to create a unique filename
                generated_name = processor.generate_docker_name() + ".png"
                output_path = os.path.join(output_dir, generated_name)
            else:
                # Use the original filename
                output_path = os.path.join(output_dir, fname)

            # Process the image using the appropriate processor
            success = processor.process_image(input_path, output_path, prompt)
            if success:
                main_pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Process PNG files with FLUX.")
    parser.add_argument('path', type=str, 
                        help='The path of the directory or file to process')

    parser.add_argument('--acceleration', '-a', type=str,
                        choices=['alimama', 'hyper', 'none'],
                        default='none',
                        help='Acceleration LORA. Available options are Alimama Turbo or ByteDance Hyper (alimama|hyper) with 10 steps. If not provided, flux with 25 steps will be used.')
    
    parser.add_argument('--prompt', '-p', type=str,
                        default='Very detailed, masterpiece quality',
                        help='Set a custom prompt, if not defined defaults to "Very detailed, masterpiece quality"')
    
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Optional output directory. If not provided, outputs will be placed in current directory.')

    # Add the new generate-names option
    parser.add_argument('--generate-names', '-g', action='store_true',
                        help='Generate unique output filenames using generate_docker_name instead of using input filenames')

    args = parser.parse_args()

    # Ensure `path` is either a file or directory
    if not os.path.exists(args.path):
        print(f"Error: {args.path} does not exist.")
        exit(1)

    # Determine output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.getcwd()  # Default to current directory

    print(f"Output directory: {out_dir}")
    print(f"Generate names: {args.generate_names}")

    process_directory(args.path, out_dir, args.acceleration, args.prompt, args.generate_names)

if __name__ == "__main__":
    main()