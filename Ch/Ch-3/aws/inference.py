import importlib
import sys
import os
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


import torch
import diffusers
import transformers
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from ip_adapter import CSGO
from diffusers.utils import load_image

import json
import io, base64
from huggingface_hub import login
import boto3
import gzip

from tqdm import tqdm
from transparent_background import Remover
from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
logger.info(f"QUAD_REMESH_AVAILABLE: {QUAD_REMESH_AVAILABLE}, TRIANGLE_REMESH_AVAILABLE: {TRIANGLE_REMESH_AVAILABLE}")
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, get_device, remove_background
import trimesh
output_dir = "stable-point-aware-3d/output"
os.makedirs(output_dir, exist_ok=True)

# Load the model and pipeline
def model_fn(model_dir, hf_token=None, aws_access_key=None, aws_secret_access_key=None, aws_region='us-west-2', context=None):
    # Set SDK socket timeout to 180 seconds
    import botocore.config
    config = botocore.config.Config(
        read_timeout=30,
        connect_timeout=30,
        retries={'max_attempts': 10}
    )

    logger.info("========== Entered model_fn ==========")
    logger.info(f"Diffusers version: {diffusers.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")

    # Logging to Hugging Face
    if hf_token is not None:
        logger.info("Logging to Hugging Face...")
        login(token=hf_token)

    csgo_device = torch.device("cuda:1") # if torch.cuda.is_available() else "cpu")
    csgo_ckpt = "/opt/ml/CSGO/CSGO/csgo_4_32.bin"
    weight_dtype = torch.float16

    logger.info("Loading VAE...")
    # pretrained_vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
    pretrained_vae_name_or_path = "/opt/ml/CSGO/base_models/sdxl-vae-fp16-fix"
    vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path,torch_dtype=torch.float16)

    logger.info("Loading controlnet...")
    controlnet_path = "/opt/ml/CSGO/base_models/TTPLanet_SDXL_Controlnet_Tile_Realistic/" 
    controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                                torch_dtype=torch.float16,
                                                use_safetensors=True)

    logger.info("Loading Stable diffusion XL...")
    sdxl_path = "/opt/ml/CSGO/base_models/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                                                        sdxl_path, #"stabilityai/stable-diffusion-xl-base-1.0",
                                                        controlnet=controlnet,
                                                        torch_dtype=torch.float16,
                                                        add_watermarker=False,
                                                        vae=vae
                                                    )
    pipe.enable_vae_tiling()
    pipe.to(csgo_device)

    logger.info("Stable Diffusion Model loaded!")

    target_content_blocks = BLOCKS['content']
    target_style_blocks = BLOCKS['style']
    controlnet_target_content_blocks = controlnet_BLOCKS['content']
    controlnet_target_style_blocks = controlnet_BLOCKS['style']

    image_encoder_path = "/opt/ml/CSGO/base_models/IP-Adapter/sdxl_models/image_encoder"

    logger.info("Creating CSGO...")
    csgo = CSGO(pipe, image_encoder_path, csgo_ckpt, csgo_device,
                num_content_tokens=4,
                num_style_tokens=32,
                target_content_blocks=target_content_blocks,
                target_style_blocks=target_style_blocks,
                # controlnet=False,
                controlnet_adapter=True,
                controlnet_target_content_blocks=controlnet_target_content_blocks,
                controlnet_target_style_blocks=controlnet_target_style_blocks,
                content_model_resampler=True,
                style_model_resampler=True,
                # load_controlnet=False,
            )
    
    # Load SPAR3D
    logger.info("Loading SPAR3D...")
    spar3d_device = "cuda:0" #torch.device("cuda:0")
    spar3d_path = "/opt/ml/stable-point-aware-3d/spar3d/models/SPAR3D_model"
    spar3d_model = SPAR3D.from_pretrained(
                            spar3d_path, #"stabilityai/stable-point-aware-3d",
                            config_name="config.yaml",
                            weight_name="model.safetensors",
                            )
    spar3d_model.to(spar3d_device)
    spar3d_model.eval()

    bg_remover = Remover(device=spar3d_device)

    # Load Bedrock
    logger.info("Loading Bedrock runtime...")
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', config=config, region_name=aws_region)
    except Exception as e:
        logger.info("Unable to Instantiate Bedrock without AWS Credentials. --> Adding Credentials...")
        bedrock_runtime = boto3.client('bedrock-runtime',
                                   aws_access_key_id=aws_access_key,
                                   config=config,
                                   aws_secret_access_key=aws_secret_access_key,
                                   region_name=aws_region)
        logger.info("Bedrock runtime instantiated!")

    return {"csgo": csgo, 
            "spar3d_model": spar3d_model,
            "bg_remover": bg_remover,
            "bedrock_runtime": bedrock_runtime}

# Process incoming input
def input_fn(serialized_input_data, content_type):
    logger.info("========== Entered input_fn ==========")
    if content_type == "application/json" or content_type == "application/json; verbose=true":
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Perform inference
def predict_fn(input_data, model):
    logger.info("========== Entered predict_fn ==========")

    logger.info("Parsing input data")
    # Parse input data
    content_image_base64 = input_data["content_image"]
    style_image_base64 = input_data["style_image"]

    image_prompt = input_data.get("image_prompt", None)
    llm_prompt = input_data.get("llm_prompt", None)
    extra_image_prompt = input_data.get("extra_image_prompt", ",  3D, 4k, highres")

    sdxl_negative_prompt = input_data.get("sdxl_negative_prompt", "lowres, blurry")
    sdxl_steps = input_data.get("sdxl_steps", 50)
    sdxl_guidance_scale = input_data.get("sdxl_guidance_scale", 10.0)
    sdxl_controlnet_conditioning_scale = input_data.get("controlnet_conditioning_scale", 0.5)
    sdxl_seed = input_data.get("sdxl_seed", 67100) # 67354
    
    sd35_seed = input_data.get("sd35_seed", 754640521) # 74564123 # 456165789
    sd35_strength = input_data.get("sd35_strength", 0.675)
    sd35_negative_prompt = input_data.get("sd35_negative_prompt", "lowres, blurry, 2D, sketch, drawing, uniform background")
    # max_sequence_length = input_data.get("max_sequence_length", 2048)

    reduction_count_type=input_data.get("reduction_count_type", "keep")
    target_count=input_data.get("target_count", 2000)
    foreground_ratio=input_data.get("foreground_ratio", 1.3)
    batch_size=input_data.get("batch_size", 1)
    texture_resolution=input_data.get("texture_resolution", 1024)
    remesh_option=input_data.get("remesh_option", 'none')

    # Convert base64 to image
    if style_image_base64 is not None:
        # Decode image from Base64
        style_image_data = base64.b64decode(style_image_base64)
        # Convert base64 to tensor if needed
        # image = torch.tensor(image)  # Example
        style_image = Image.open(io.BytesIO(style_image_data)).convert("RGB")#.resize((512,512))
    else:
        raise ValueError("Input must include a style image ")
    
    if content_image_base64 is not None:
        # Decode image from Base64
        content_image_data = base64.b64decode(content_image_base64)
        # Convert base64 to tensor if needed
        # image = torch.tensor(image)  # Example
        content_image = Image.open(io.BytesIO(content_image_data)).convert("RGB")#.resize((512,512))
    else:
        raise ValueError("Input must include a content image ")

    #  Parse model
    csgo = model["csgo"]
    bedrock_runtime = model["bedrock_runtime"]

    if image_prompt is not None:
        logger.info("Image prompt provided, bypassing call to LLM.")
        prompt = image_prompt
    else:
        logger.info("No image prompt provided, calling LLM to generate prompt.")
        if llm_prompt is not None:
            logger.info("LLM prompt provided.")
            llm_text_payload = llm_prompt
        else:
            logger.info("No LLM prompt provided, using default prompt.")
            llm_text_payload = "This is a sketch. Describe what it represents in a manner that constitutes a good prompt for a text-to-image model to generate a realistic photo of the contents of the sketch. Only output the prompt text without any additional wrapping text. Use that bare-bone prompt and embellish it a bit to generate an interesting image. Add elements of language to generate an image with obvious 3D features and remove any sketch-like elements."

        if content_image is not None:
            # generate payload for bedrock
            payload_with_image = {
                                    "anthropic_version": "bedrock-2023-05-31",
                                    "max_tokens": 200,
                                    "top_k": 250,
                                    "stop_sequences": [],
                                    "temperature": 1,
                                    "top_p": 0.999,
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": llm_text_payload},
                                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": content_image_base64}}
                                            ]
                                        }
                                    ]
                                }
            # Invoke the model with image and text
            response = bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload_with_image),
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            generated_text = response_body['content'][0]['text']
            logger.info(f"LLM Generated text: {generated_text}")

        else:
            logger.info("No image provided, cannot generate prompt.")
            generated_text = ""

        prompt = generated_text
    logger.info(f"Prompt for image generation: {prompt}")

    # Generate image
    logger.info("Generating image")
    # CSGO inference
    num_sample = 1
    better_sketch_image = csgo.generate(pil_content_image= content_image, pil_style_image=style_image,
                           prompt=prompt,
                           negative_prompt=sdxl_negative_prompt,
                           content_scale=1.0,
                           style_scale=1.0,
                           guidance_scale=sdxl_guidance_scale,
                           num_images_per_prompt=num_sample,
                           num_samples=1,
                           num_inference_steps=sdxl_steps,
                           seed=sdxl_seed, # 4657,
                           image=content_image.convert('RGB'),
                           controlnet_conditioning_scale=sdxl_controlnet_conditioning_scale,
                          )[0]

    logger.info("Better sketch generated")

    # Feed the better sketch to SD3.5 and generate the final image
    logger.info("Encoding better sketch to base64")
    encoded_sketch = encode_image_base64(better_sketch_image)

    logger.info("Generating final image")
    model_id = "stability.sd3-5-large-v1:0"  # Replace with the Bedrock model you want to use
    prompt = prompt + " " + extra_image_prompt
    if sd35_negative_prompt is not None:
        sd35_input_payload = {
            "prompt": prompt,
            "negative_prompt": sd35_negative_prompt,
            "image": encoded_sketch,
            "mode": "image-to-image",
            "output_format":"jpeg",
            "strength":sd35_strength,
            "seed": sd35_seed
        }
    else:
        sd35_input_payload = {
            "prompt": prompt,
            "image": encoded_sketch,
            "mode": "image-to-image",
            "output_format":"jpeg",
            "strength":sd35_strength,
            "seed": sd35_seed
        }
    sd35_response = bedrock_runtime.invoke_model(
                                    modelId=model_id,
                                    contentType="application/json",
                                    accept="application/json",
                                    body=json.dumps(sd35_input_payload)
                                )
    logger.info("SD3.5 response received")
    sd35_response_body = json.loads(sd35_response["body"].read().decode("utf-8"))


    # Generate 3D model
    spar3d_model = model["spar3d_model"]
    bg_remover = model["bg_remover"]
    spar3d_args = Spar3d_Args(reduction_count_type=reduction_count_type,
                              target_count=target_count,
                              foreground_ratio=foreground_ratio,
                              batch_size=batch_size,
                              texture_resolution=texture_resolution,
                              remesh_option=remesh_option) # Add here ability to take inputs
    # Decode SD3.5 response
    logger.info("Decoding SD3.5 response")
    sd35_response_image = decode_image_base64(sd35_response_body['images'][0])

    logger.info("Generating 3D model")
    input_images = [sd35_response_image]
    generated_3d_mesh = generate_3d(spar3d_model, bg_remover, spar3d_args, input_images)
    logger.info("3D model generated")

    output = {
            "Image_prompt": prompt,
            "SD3.5_response": sd35_response_body,
            "sketch": encoded_sketch,
            "3D_mesh": serialize_trimesh_glb(generated_3d_mesh), #generated_3d_mesh
            }

    return output #sd35_response_body

class Spar3d_Args:
    def __init__(self, reduction_count_type="keep",
                target_count=2000,
                foreground_ratio=1.3,
                batch_size=1,
                texture_resolution=1024,
                remesh_option='none'
                ):
        self.reduction_count_type = reduction_count_type if reduction_count_type in ['keep', 'vertex', 'faces'] else 'keep'
        self.target_count = target_count if target_count > 0 else 2000
        self.foreground_ratio = foreground_ratio
        self.batch_size = batch_size
        self.texture_resolution = texture_resolution
        self.remesh_option = remesh_option if remesh_option in ['none', 'triangle', 'quad'] else 'none'
        logger.info(f"Mesh options: \n reduction_count_type: {self.reduction_count_type}, \ntarget_count: {self.target_count}, \n foreground_ratio: {self.foreground_ratio}, \n batch_size: {self.batch_size}, \n texture_resolution: {self.texture_resolution}, \n remesh_option: {self.remesh_option}")

def serialize_trimesh_glb(mesh):
    """Convert a Trimesh object to base64-encoded .glb"""
    buffer = io.BytesIO()
    mesh.export(buffer, file_type='glb')  # Export as GLB format
    encoded_mesh = base64.b64encode(buffer.getvalue()).decode('utf-8')
    size_in_bytes = len(encoded_mesh)
    logger.info(f"Mesh GLB Size: {size_in_bytes / 1024:.2f} KB")
    return encoded_mesh

def generate_3d(spar3d_model, bg_remover, spar3d_args, input_images):
    images = []
    for image in input_images:
        image = remove_background(image.convert("RGBA"), bg_remover)
        image = foreground_crop(image, spar3d_args.foreground_ratio)
        images.append(image)

    vertex_count = (
                    -1
                    if spar3d_args.reduction_count_type == "keep"
                    else (
                        spar3d_args.target_count
                        if spar3d_args.reduction_count_type == "vertex"
                        else spar3d_args.target_count // 2
                    )
                )
    
    logger.info("Generating 3D model")
    spar3d_device = spar3d_model.device.type+":"+str(spar3d_model.device.index)
    for i in tqdm(range(0, len(images), spar3d_args.batch_size)):
        image = images[i : i + spar3d_args.batch_size]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(spar3d_device, dtype=torch.bfloat16):
                mesh, _ = spar3d_model.run_image(image, 
                                          bake_resolution=spar3d_args.texture_resolution, 
                                          remesh=spar3d_args.remesh_option, 
                                          vertex_count=vertex_count, 
                                          return_points=False)

    logger.info("3D model generated")
    return mesh

def encode_image_base64(image):
    """Convert a PIL image to a base64 string and check its size."""
    if image is None:
        return None

    # logger.info("Encoding image to base64")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        buffer.seek(0)
        raw_bytes = buffer.getvalue()
        raw_size = len(raw_bytes)  # Check raw file size
        # logger.info(f"Raw PNG Size: {raw_size / 1024:.2f} KB")  # Convert to KB

        base64_string = base64.b64encode(raw_bytes).decode("utf-8")
        base64_size = len(base64_string)
        # logger.info(f"Base64 Encoded Size: {base64_size / 1024:.2f} KB")

        return base64_string

def gzip_compress(data):
    """Gzip compress a string and check its size."""
    compressed = gzip.compress(data.encode("utf-8"))
    compressed_size = len(compressed)
    logger.info(f"Gzip Compressed Size: {compressed_size / 1024:.2f} KB")
    return base64.b64encode(compressed).decode("utf-8")

def decode_image_base64(base64_string):
    """Convert a base64 string back into a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

# Format the output
def output_fn(prediction, accept):
    logger.info("========== Entered output_fn ==========")
    if accept in ["application/json", "application/json; verbose=true"]:
        logger.info("in application/json")
        # Convert image to base64
        # image_base64 = encode_image_base64(prediction)
        
        # # For async endpoint calls, return a response with additional metadata
        # response = {
        #     "image": image_base64,
        #     "status": "completed",
        #     "error": None
        # }
        return prediction # response
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
