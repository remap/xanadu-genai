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
from PIL import Image
from huggingface_hub import login
import boto3
import gzip

import gc 

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
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    
    # Decode the generated output.
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    sft_format = getattr(prepare_inputs, "sft_format", [""])[0]  # (Adjust if needed)
    
    result = answer
    if json_format:
        try:
            result = json.loads(answer)
        except Exception as e:
            raise ValueError(f"Failed to parse answer as JSON. Answer: {answer}\nError: {e}")
    
    return answer


# def generate_coordinates(background_image, 
#                          muse_image, 
#                          bg_width, 
#                          bg_height, 
#                          person_width, 
#                          person_height,
#                          vl_chat_processor, tokenizer, vl_gpt) -> dict:
#     """
#     Given the background and person images along with their dimensions,
#     returns a dictionary with the recommended placement parameters:
#       - top_offset (in pixels)
#       - left_offset (in pixels)
#       - scale_factor (a float; e.g., 0.5 means scale the person image to 50% of its original height)
    
#     This function uses a VL model (via Deepseek) to determine where to place the person image.
#     """
#     # Define a prompt that instructs the model to output placement parameters in JSON.
#     prompt = (
#         "Given the following image dimensions:\n"
#         f"Background: width={bg_width}, height={bg_height}.\n"
#         f"Person: width={person_width}, height={person_height}.\n"
#         "Based on the scene in the background and the person's pose, "
#         "please determine where the person should be placed on the background. "
#         "Output the recommended placement as a JSON object with the keys "
#         "'top_offset', 'left_offset', and 'scale_factor'. The 'top_offset' and "
#         "'left_offset' should indicate the pixel coordinates for where the top-left corner "
#         "of the person image should be pasted on the background, and 'scale_factor' "
#         "should be a float representing the factor by which to scale the person image."
#     )

#     # Generate placement parameters using Deepseek.
#     answer = generate_prompt([background_image, muse_image], 
#                              vl_chat_processor, tokenizer, vl_gpt, 
#                              prompt=prompt, json_format=True)
    
#     return answer

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

    # Load Bedrock
    # logger.info("Loading Bedrock runtime...")
    # bedrock_runtime = boto3.client('bedrock-runtime',
    #                                aws_access_key_id=aws_access_key,
    #                                aws_secret_access_key=aws_secret_access_key,
    #                                region_name=aws_region)

    seg_model = tvmodels.segmentation.deeplabv3_resnet101(pretrained=True)
    seg_model.to('cuda:3').eval()
    logger.info("Segmentation model loaded!")

    seg_transform = tvtransforms.Compose( [
                                            tvtransforms.Resize(520),
                                            tvtransforms.ToTensor(),
                                            tvtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                            ] )

    return {
            "sd_pipe": sd_pipe, 
            # "bedrock_runtime": bedrock_runtime,
            "vl_chat_processor": vl_chat_processor,
            "tokenizer": tokenizer,
            "vl_gpt": vl_gpt,
            "seg_model": seg_model,
            "seg_transform": seg_transform
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
    #image_prompt = input_data.get("image_prompt", None)
    #extra_image_prompt = input_data.get("extra_image_prompt", " in a realistic 1980s retro vaporwave style")
    #background_generationprompt = input_data.get("llm_prompt", None)
    negative_prompt = input_data.get("negative_prompt", "")
    steps = input_data.get("steps", 50)
    guidance_scale = input_data.get("guidance_scale", 7.5)
    controlnet_condition_scale = input_data.get("controlnet_condition_scale", 0.8)
    generator_seed = input_data.get("generator_seed", 0)
    background_prompt = input_data.get("background_prompt", "Analyze the given background sketch and provide a detailed, photorealistic description of the scene. Describe every element as it would appear in real life, including natural lighting, textures, colors, and spatial relationships between objects. Do not reference any sketch-like or hand-drawn qualities; instead, focus solely on creating a description that translates the sketch into a fully realistic rendering. Include details such as the environment's mood, shadow directions, reflective surfaces, and any subtle variations in tone, ensuring that the resulting description can be used as a reference for creating a true-to-life background scene.")
    negative_prompt = input_data.get("negative_prompt", "lowres, blurry, 2D, sketch, drawing, uniform background")
    # spreadsheet_url = input_data.get("params_url", "https://docs.google.com/spreadsheets/d/1EIe9OFpqO1Wc0sYjNSKXhSRBO7k17v9wZE34b3F3TJ4/edit?gid=866270832#gid=866270832")
    # # max_sequence_length = input_data.get("max_sequence_length", 2048)

    # params_dict = get_param_dict(spreadsheet_url)
    # logger.info(f"Params dict: {params_dict}")

    # # Retrieving relevant prompts from param_dict
    # background_description_prompt = params_dict["ch1-default-background_description_prompt"]
    # vaporwave_prompt = params_dict["ch1-default-80s_prefix"]
    # negative_prompt = params_dict["ch1-default-80s_negative"]

    # Convert base64 to image
    if bg_image_base64 is not None:
        # Decode image from Base64
        bg_image_data = base64.b64decode(bg_image_base64)
        bg_image = Image.open(io.BytesIO(bg_image_data)).convert("RGB")#.resize((512,512))
    else:
        raise ValueError("Input must include a background image.")
    
    if muse_image_base64 is not None:
        # Decode image from Base64
        muse_image_data = base64.b64decode(muse_image_base64)
        muse_image = Image.open(io.BytesIO(muse_image_data)).convert("RGB")#.resize((512,512))
    else:
        raise ValueError("Input must include a muse image.")


    #  Parse model
    sd_pipe = model["sd_pipe"]
    # bedrock_runtime = model["bedrock_runtime"]
    vl_chat_processor = model["vl_chat_processor"]
    tokenizer = model["tokenizer"]
    vl_gpt = model["vl_gpt"]

    # Generate prompt
    #background_description_prompt = params_dict["ch1-default-background_description_prompt"]
    logger.info("Generating background description")
    background_description = generate_prompt(background_prompt, vl_chat_processor, tokenizer, vl_gpt, image = bg_image)
    logger.info(f"Background description generated: {background_description}")

    # Generate image
    logger.info("Generating image")
    # Stable Diffusion inference
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    prompt = "Using the provided control image (the original background sketch) as a guide, generate a fully realistic rendering of the scene described below. Focus on achieving lifelike lighting, textures, and colors that translate the sketch into a natural, high-resolution image. Follow this detailed description precisely: " + background_description + ". Ensure the final output has no sketch-like qualities, but instead looks like a real-world photograph with accurate shadows, depth, and material details."
    generated_image = sd_pipe(
        prompt = prompt,
        negative_prompt=negative_prompt,
        control_image=bg_image,
        controlnet_conditioning_scale=controlnet_condition_scale,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]#.resize((512, 512))

    logger.info("Background image generated")

    # Compositing in Muse
    logger.info("Compositing in Muse")
    to_tensor = tvtransforms.ToTensor()
    to_pil = tvtransforms.ToPILImage() 
    bg_tensor = to_tensor(generated_image)
    person_tensor = to_tensor(muse_image)

    # logger.info("Loading segmentation model")
    # seg_model = tvmodels.segmentation.deeplabv3_resnet101(pretrained = True)
    # seg_model.eval()

    # seg_transform = tvtransforms.Compose( [
    #                                         tvtransforms.Resize(520),
    #                                         tvtransforms.ToTensor(),
    #                                         tvtransforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                     std=[0.229, 0.224, 0.225])
    #                                         ] )

    # get segmentation model
    seg_model = model["seg_model"]
    seg_transform = model["seg_transform"]

    seg_model_device = next(seg_model.parameters()).device
    person_for_seg = seg_transform(muse_image).unsqueeze(0).to(seg_model_device)
    logger.info("Segmenting person")
    with torch.no_grad():
        output = seg_model(person_for_seg)['out'][0].cpu()

    logger.info("Generating mask")
    person_class = 15 # typically the case for COCO
    mask_pred = output.argmax(0) == person_class # boolean mask
    mask = mask_pred.to(torch.float32)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                         size = person_tensor.shape[1:], 
                         mode = 'bilinear', 
                         align_corners = False).squeeze()
    # Composite
    # Expand mask to have same number of channels as person image
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and person_tensor.shape[0] == 3:
        mask = mask.expand(3, -1, -1)

    logger.info("Determine placement")
    # Determine desired scale relative to the background.
    bg_height, bg_width = bg_tensor.shape[1], bg_tensor.shape[2]
    scale_factor = 0.5  # For example, set the person height to 50% of background height.
    new_height = int(bg_height * scale_factor)
    person_height, person_width = person_tensor.shape[1], person_tensor.shape[2]
    new_width = int(person_width * new_height / person_height)  # maintain aspect ratio
    # Get new tensors and masks
    person_tensor_small = F.interpolate(person_tensor.unsqueeze(0),
                                        size=(new_height, new_width),
                                        mode='bilinear',
                                        align_corners=False).squeeze(0)
    mask_small = F.interpolate(mask.unsqueeze(0),
                            size=(new_height, new_width),
                            mode='bilinear',
                            align_corners=False).squeeze(0)
    mask_small = torch.clamp(mask_small, 0, 1)
    bg_h, bg_w = bg_tensor.shape[1], bg_tensor.shape[2]
    person_h, person_w = person_tensor.shape[1], person_tensor.shape[2]

    def get_placement_offsets_bottom_center(background, person):
        """
        Compute offsets to place the person at the bottom center of the background.
        :param background: Tensor of shape [C, H, W] for the background.
        :param person: Tensor of shape [C, h, w] for the scaled person.
        :return: (top_offset, left_offset) in pixel coordinates.
        """
        bg_h, bg_w = background.shape[1], background.shape[2]
        person_h, person_w = person.shape[1], person.shape[2]
        top_offset = bg_h - person_h  # Align bottom edges.
        left_offset = (bg_w - person_w) // 2  # Center horizontally.
        return top_offset, left_offset
    
    logger.info("Get placement offsets")
    top_offset, left_offset = get_placement_offsets_bottom_center(bg_tensor, person_tensor_small)
    if top_offset < 0 or left_offset < 0 or (top_offset + new_height > bg_height) or (left_offset + new_width > bg_width):
        raise ValueError("The computed offsets cause the person image to fall outside the background.")
    logger.info(f"Placement offsets: top={top_offset}, left={left_offset}")

    logger.info("Compositing")
    composite = bg_tensor.clone()
    bg_roi = composite[:, top_offset:top_offset+new_height, left_offset:left_offset+new_width] # getting new position
    blended_roi = person_tensor_small * mask_small + bg_roi * (1 - mask_small) # getting composite 
    composite[:, top_offset:top_offset+new_height, left_offset:left_offset+new_width] = blended_roi
    composite_img = to_pil(composite)
    logger.info("Image compositing done")

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
