import importlib
import sys
import os
import logging
import requests # this should be imported into all modules

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


import torch
import diffusers
import transformers

from transformers import AutoModelForCausalLM
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

import torch.nn.functional as F
import torchvision.transforms as tvtransforms
import torchvision.models as tvmodels

import json
import io, base64
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from huggingface_hub import login
import boto3
import gzip
import random
import cv2

import gc 
#from RealESRGAN import RealESRGAN
from rembg import remove
import numpy as np

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)
    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image
    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image
    
def generate_prompt(prompt, vl_chat_processor, tokenizer, vl_gpt, image=None, json_format=False) -> str:
    # Set a default prompt if none is provided.
    if prompt is None:
        prompt = ("Analyze the given background sketch and provide a detailed, photorealistic description of the scene. Describe every element as it would appear in real life, including natural lighting, textures, colors, and spatial relationships between objects. Do not reference any sketch-like or hand-drawn qualities; instead, focus solely on creating a description that translates the sketch into a fully realistic rendering. Include details such as the environment's mood, shadow directions, reflective surfaces, and any subtle variations in tone, ensuring that the resulting description can be used as a reference for creating a true-to-life background scene. ")
    
    # Build the conversation using the provided prompt and image.
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder> {prompt}",
            "images": [image] if image is not None else []
        },
        {"role": "Assistant", "content": ""},
    ]
    
    # Load the image(s) for this conversation.
    if image is not None and isinstance(image, Image.Image):
        pil_images = [image]
    else:
        pil_images = load_pil_images(conversation)
    
    # Prepare inputs using the VLChat processor.
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    
    # # Ensure all tensor attributes are moved to the GPU.
    # for key, value in vars(prepare_inputs).items():
    #     if isinstance(value, torch.Tensor):
    #         setattr(prepare_inputs, key, value.cuda())

    # setattr(prepare_inputs, key, value.cuda(vl_gpt.device))
    
    # Run the image encoder to get image embeddings.
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    # Generate a response using the language model.
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )
    
    # Decode the generated output.
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    max_text_length = 980
    if len(answer) > max_text_length:
        answer = answer[:max_text_length].rstrip()

    return answer


# Load the model and pipeline
def model_fn(model_dir, hf_token, aws_access_key=None, aws_secret_access_key=None, aws_region='us-west-2', context=None):

    logger.info("========== Entered model_fn ==========")
    logger.info(f"Diffusers version: {diffusers.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")

    # Logging to Hugging Face
    logger.info("Logging to Hugging Face...")
    login(token=hf_token)

    logger.info("Loading controlnet...")
    controlnet = SD3ControlNetModel.from_pretrained("stabilityai/stable-diffusion-3.5-large-controlnet-canny",
                                                torch_dtype=torch.float16)
    logger.info("Loading Stable diffusion 3.5 large...")
    sd_pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                                                            "stabilityai/stable-diffusion-3.5-large",
                                                            controlnet=controlnet,
                                                            torch_dtype=torch.float16,
                                                            device_map="balanced",
                                                            max_memory={0: "24GB", 1: "24GB", 2: "24GB", 3: "24GB"},
                                                        )
    #controlnet.to('cuda:3')
    sd_pipe.controlnet.to('cuda:3')
    sd_pipe.image_processor = SD3CannyImageProcessor()
    sd_pipe.load_ip_adapter(
        "InstantX/SD3.5-Large-IP-Adapter",  
        subfolder="",                       
        weight_name="ip-adapter.bin", 
        image_encoder_folder="google/siglip-so400m-patch14-384"# exact filename
    )
    logger.info("Stable Diffusion Model loaded!")

    # Load DeepSeek
    logger.info("Loading DeepSeek model...")
    # Specify the model identifier.
    model_path = "deepseek-ai/deepseek-vl-7b-chat"

    # Load the VLChat processor and tokenizer.
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the model.
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                                                    model_path, trust_remote_code=True
                                                )
    # Move model to GPU (and remove bfloat16 if it causes issues).
    vl_gpt.to(torch.bfloat16).to("cuda:2").eval()
    logger.info(f"vl gpt device: {vl_gpt.device}; vl gpt dtype: {vl_gpt.dtype}")
    logger.info("DeepSeek Model loaded!")

    #Load Bedrock
    logger.info("Loading Bedrock runtime...")
    bedrock_runtime = boto3.client('bedrock-runtime',
                                   aws_access_key_id=aws_access_key,
                                   aws_secret_access_key=aws_secret_access_key,
                                   region_name='us-east-1')

    # seg_model = tvmodels.segmentation.deeplabv3_resnet101(pretrained=True)
    # seg_model.to('cuda:3').eval()
    # logger.info("Segmentation model loaded!")

    # seg_transform = tvtransforms.Compose( [
    #                                         tvtransforms.Resize(520),
    #                                         tvtransforms.ToTensor(),
    #                                         tvtransforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                     std=[0.229, 0.224, 0.225])
    #                                         ] )
    # ESRGAN model
    # model_dir = '/opt/ml/Real-ESRGAN/weights'
    # model_name = 'RealESRGAN_x4.pth'
    # gan_model = RealESRGAN('cuda:1', scale=4)
    # gan_model.load_weights(os.path.join(model_dir, model_name), download=False)

    return {
            "sd_pipe": sd_pipe, 
            "bedrock_runtime": bedrock_runtime,
            "vl_chat_processor": vl_chat_processor,
            "tokenizer": tokenizer,
            "vl_gpt": vl_gpt,
            # "seg_model": seg_model,
            # "seg_transform": seg_transform,
            # "gan_model": gan_model
            }

# Process incoming input
def input_fn(serialized_input_data, content_type):
    logger.info("========== Entered input_fn ==========")
    if content_type == "application/json":
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Perform inference
def predict_fn(input_data, model):
    logger.info("========== Entered predict_fn ==========")
    # Extract inputs
    # prompt = input_data.get("prompt", "")
    # image = input_data.get("image", None)  # Optional: Can process ControlNet image

    logger.info("Parsing input data")
    # Parse input data
    bg_image_base64 = input_data["bg_image"]
    muse_image_base64 = input_data["muse_image"]
    ref_image_base64 = input_data["ref_image"]
    #image_prompt = input_data.get("image_prompt", None)
    muse_prompt = input_data.get("muse_prompt", "")
    #background_generationprompt = input_data.get("llm_prompt", None)
    steps = input_data.get("steps", 50)
    guidance_scale = input_data.get("guidance_scale", 5)
    controlnet_condition_scale = input_data.get("controlnet_condition_scale", 0.4)
    generator_seed = input_data.get("generator_seed", 0)
    ip_adapter_scale = input_data.get("ip_adapter_scale", 0.6)
    llm_prompt = input_data.get("llm_prompt", "Analyze the given background sketch and provide a detailed, photorealistic description of the scene. Describe every element as it would appear in real life, including natural lighting, textures, colors, and spatial relationships between objects. Do not reference any sketch-like or hand-drawn qualities; instead, focus solely on creating a description that translates the sketch into a fully realistic rendering. Include details such as the environment's mood, shadow directions, reflective surfaces, and any subtle variations in tone, ensuring that the resulting description can be used as a reference for creating a true-to-life background scene.")
    negative_prompt = input_data.get("negative_prompt", "lowres, blurry, 2D, sketch, drawing, uniform background")
    background_description = input_data.get("background_description", None)
    # Convert base64 to image
    if bg_image_base64 is not None:
        # Decode image from Base64
        bg_image_data = base64.b64decode(bg_image_base64)
        bg_image = Image.open(io.BytesIO(bg_image_data)).convert("RGB")#.resize((512,512))
        logger.info(f" Background image mode: {bg_image.mode}")
    else:
        raise ValueError("Input must include a background image.")
    
    if muse_image_base64 is not None:
        # Decode image from Base64
        muse_image_data = base64.b64decode(muse_image_base64)
        muse_image = Image.open(io.BytesIO(muse_image_data)).convert("RGBA")#.resize((512,512)) #RGBA so image is transparent
        logger.info(f" Muse image mode: {muse_image.mode}")
    else:
        raise ValueError("Input must include a muse image.")
    
    if ref_image_base64 is not None:
        # Decode image from Base64
        ref_image_data = base64.b64decode(ref_image_base64)
        ref_image = Image.open(io.BytesIO(ref_image_data)).convert("RGB")#.resize((512,512))
    else:
        raise ValueError("Input must include a reference image.")


    #  Parse model
    sd_pipe = model["sd_pipe"]
    bedrock_runtime = model["bedrock_runtime"]
    vl_chat_processor = model["vl_chat_processor"]
    tokenizer = model["tokenizer"]
    vl_gpt = model["vl_gpt"]

    # Generate prompt
    logger.info("Generating background description")
    if background_description == "nan":
        logger.info("Background description not provided, generating using LLM")
        background_description = generate_prompt(llm_prompt, vl_chat_processor, tokenizer, vl_gpt, image = bg_image)
    logger.info(f"Background description generated: {background_description}")

    # model prompt
    model_prompt = "Using the provided control image (the original background sketch) as a guide, generate a fully realistic rendering of the scene described below. Focus on achieving lifelike lighting, textures, and colors that translate the sketch into a natural, high-resolution image. Follow this detailed description precisely: " + background_description + ". Ensure the final output has no sketch-like qualities, but instead looks like a real-world photograph with accurate shadows, depth, and material details."
    logger.info(f"Model prompt: {model_prompt}")
    # Generate image
    logger.info("Generating image")
    # Stable Diffusion inference
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    #prompt = "Using the provided control image (the original background sketch) as a guide, generate a fully realistic rendering of the scene described below. Focus on achieving lifelike lighting, textures, and colors that translate the sketch into a natural, high-resolution image. Follow this detailed description precisely: " + background_description + ". Ensure the final output has no sketch-like qualities, but instead looks like a real-world photograph with accurate shadows, depth, and material details."
    sd_pipe.set_ip_adapter_scale(ip_adapter_scale)
    prompt = model_prompt
    if muse_prompt != "nan":
        # Add the muse prompt to the model prompt
        logger.info(f"Muse prompt: {muse_prompt}")
        prompt = model_prompt + muse_prompt
    original_image = sd_pipe(
        prompt = prompt,
        negative_prompt=negative_prompt,
        control_image=bg_image,
        ip_adapter_image = ref_image,
        controlnet_conditioning_scale=controlnet_condition_scale,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width = 512,
        height = 384,
        generator=generator,
    ).images[0]

    logger.info("Background image generated")


    # call Amazon Nova Canvas for resizing + super resolution
    logger.info("Calling Nova Canvas…")
    buf = io.BytesIO()
    original_image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 2. Build the IMAGE_VARIATION payload
    variation_body = json.dumps({
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": {
            "text": background_description + "photorealistic,cinematic,8k,hdr",
            "negativeText": "bad quality, low resolution, cartoon",
            "images": [ img_b64 ],          
            "similarityStrength": 0.7       
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 2048,
            "cfgScale": 6.5     
        }
    })

    # 3. Invoke Nova Canvas IMAGE_VARIATION
    response = bedrock_runtime.invoke_model(
        modelId="amazon.nova-canvas-v1:0",
        body=variation_body,
        contentType="application/json",
        accept="application/json"
    )
    logger.info("Nova image generated")

    # 4. Decode the returned image
    resp_json   = json.loads(response["body"].read())
    out_b64     = resp_json["images"][0]
    out_bytes   = base64.b64decode(out_b64)
    generated_image = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    logger.info("decoding of Nova image generated")

    def resize_image(image, target_size=(1024, 256)):
            
        if image.size != target_size:
            image_np = cv2.resize(np.array(image), target_size, interpolation=cv2.INTER_AREA)
            #image_np = cv2.edgePreservingFilter(image_np, flags=1, sigma_s=20, sigma_r=0.07)
            image = Image.fromarray(image_np)
            
        return image
    
    # Resize the image
    # generated_image = resize_image(generated_image, target_size=(1024, 192))
    generated_image = generated_image.resize((1024, 192), resample=Image.LANCZOS)
    logger.info("Background image resized")

    # Compositing in Muse
    logger.info("Compositing in Muse")
    # logger.info(" Converting muse image from RGB to RGBA")
    # # Convert the muse image to RGBA
    # muse_np = np.array(muse_image)
    # muse_transparent_np = remove(muse_np)   # returns an RGBA numpy array
    muse_rgba = muse_image #Image.fromarray(muse_transparent_np)    
    logger.info(f" Muse image mode after conversion: {muse_rgba.mode}")
    # Convert the background image to RGBA
    # logger.info(" Converting generated image from RGB to RGBA")
    bg_rgba = generated_image#.convert("RGBA")  
    logger.info(f" Background image mode after conversion: {bg_rgba.mode}")

    # muse_rgba = muse_image
    # logger.info(f" Muse image mode after conversion: {muse_rgba.mode}")
    # logger.info(f" Background image mode after conversion: {bg_rgba.mode}")
    # Convert the muse image to a tensor
    # logger.info(" Converting generated image to tensor")
    # to_tensor = tvtransforms.ToTensor()
    # to_pil = tvtransforms.ToPILImage() 
    # bg_tensor = to_tensor(generated_image)
    # person_tensor = to_tensor(muse_rgba)
    # bg_rgba = generated_image
    # muse_rgba = muse_image

    def get_random_bottom_offset(bg_size, fg_size, margin=150):
        bg_w, bg_h = bg_size
        fg_w, fg_h = fg_size
        top      = bg_h - fg_h
        max_left = bg_w - fg_w - margin
        left     = (bg_w - fg_w)//2 if max_left < margin else random.randint(margin, max_left)
        return left, top

    def composite_muse_image(
        background: Image.Image,
        muse: Image.Image,
        max_h_pct: float = 0.8,
        max_w_pct: float = 0.5,
    ) -> Image.Image:
        """
        Composite a muse image onto a background image.
        """
        bg = background
        mu = muse
        logger.info("Beginning to composite muse image")
        # scale muse
        bg_w, bg_h = bg.size
        m_w, m_h   = mu.size
        scale      = min((bg_h * max_h_pct)/m_h, (bg_w * max_w_pct)/m_w)
        nw, nh     = int(m_w * scale), int(m_h * scale)
        mu_small   = mu.resize((nw, nh), Image.LANCZOS)

        # pick random bottom‐aligned spot
        left, top = get_random_bottom_offset((bg_w, bg_h), (nw, nh))

        # # get silhouette mask, dilate & blur
        # silh = mu_small.split()[3]                                     # pure alpha
        # silh = silh.filter(ImageFilter.MaxFilter(dilate_px*2+1))       # dilate
        # silh = silh.filter(ImageFilter.GaussianBlur(blur_radius))     # soft edge

        # # crop & brighten exactly under that shape
        # region = (left, top, left+nw, top+nh)
        # patch  = bg.crop(region)
        # bright = ImageEnhance.Brightness(patch).enhance(brighten_factor)
        # # paste brightened patch *using the silh mask* instead of a rectangle
        # bg.paste(bright, (left, top), silh)

        # paste the muse itself on top (using its alpha)
        bg.paste(mu_small, (left, top), mu_small)
        #bg.alpha_composite(mu_small, (left, top))
        logger.info("Finished compositing muse image")
        return bg#.convert("RGB")

    composite_img = composite_muse_image(
        bg_rgba,
        muse_rgba,
        max_h_pct=0.8,
        max_w_pct=0.5,
    )
    logger.info("Muse image composited")

    return composite_img

def encode_image_base64(image):
    """Convert a PIL image to a base64 string and check its size."""
    if image is None:
        return None

    logger.info("Encoding image to base64")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        buffer.seek(0)
        raw_bytes = buffer.getvalue()
        raw_size = len(raw_bytes)  # Check raw file size
        logger.info(f"Raw PNG Size: {raw_size / 1024:.2f} KB")  # Convert to KB

        base64_string = base64.b64encode(raw_bytes).decode("utf-8")
        base64_size = len(base64_string)
        logger.info(f"Base64 Encoded Size: {base64_size / 1024:.2f} KB")

        return base64_string

def gzip_compress(data):
    """Gzip compress a string and check its size."""
    compressed = gzip.compress(data.encode("utf-8"))
    compressed_size = len(compressed)
    logger.info(f"Gzip Compressed Size: {compressed_size / 1024:.2f} KB")
    return base64.b64encode(compressed).decode("utf-8")

# Format the output
def output_fn(prediction, accept):
    logger.info("========== Entered output_fn ==========")
    if accept == "application/json":#"image/png":
        logger.info("in application/json")
        # Convert image to base64
        image_base64 = encode_image_base64(prediction)
        
        return {"image": image_base64}

    else:
        raise ValueError(f"Unsupported accept type: {accept}")
