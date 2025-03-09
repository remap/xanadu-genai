import os
import sys
import io
import json
import base64
import gzip
import logging
import random
import gc
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
from diffusers import AutoPipelineForInpainting





# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Helper functions ---

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def encode_image_base64(image: Image.Image) -> str:
    logger.info("Encoding image to base64")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        buffer.seek(0)
        raw_bytes = buffer.getvalue()
        logger.info(f"Raw PNG Size: {len(raw_bytes)/1024:.2f} KB")
        base64_string = base64.b64encode(raw_bytes).decode("utf-8")
        logger.info(f"Base64 Encoded Size: {len(base64_string)/1024:.2f} KB")
        return base64_string

def gzip_compress(data: str) -> str:
    compressed = gzip.compress(data.encode("utf-8"))
    logger.info(f"Gzip Compressed Size: {len(compressed)/1024:.2f} KB")
    return base64.b64encode(compressed).decode("utf-8")

def decode_base64_image(image_b64: str) -> Image.Image:
    image_data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

# --- DeepSeek Prompt Generation ---
def generate_prompt(images, vl_chat_processor, tokenizer, vl_gpt, prompt=None, json_format=False) -> str:

    ## This prompt can be played around with
    if prompt is None:
        prompt = "Give me a caption for this background scene. Please describe as many aspects of the scene as you can, with specific descriptions of what is happening in each portion of the image. Please also describe the relative locations of objects and imagery in the scene in relation to each other. "
    
    conversation = [
        {
            "role": "User",
            # Note: Prepend the placeholder if your model expects it.
            "content": f"<image_placeholder> {prompt}",
            "images": images
        },
        {"role": "Assistant", "content": ""},
    ]

    # Load the image(s) for this conversation.
    if isinstance(images[0], Image.Image):
        pil_images = images
    else:
        pil_images = load_pil_images(conversation)

    # torch.cuda.set_device(vl_gpt.device)

    # Prepare inputs using the VLChat processor.
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # If prepare_inputs does not have a .to() method for all tensors,
    # iterate over its attributes and move any tensor to GPU.
    for key, value in vars(prepare_inputs).items():
        if isinstance(value, torch.Tensor):
            setattr(prepare_inputs, key, value.cuda())

    # Run the image encoder to get image embeddings.
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate a response.
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

    # Decode the output.
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    sft_format = getattr(prepare_inputs, "sft_format", [""])[0]  # Adjust if needed.

    # Save the result.
    # result = {
    #     "prompt": prompt,
    #     "image": image,
    #     "response": answer,
    # }

    result = answer

    if json_format:
        try:
            result = json.loads(answer)
        except Exception as e:
            raise ValueError(f"Failed to parse answer as JSON. Answer: {answer}\nError: {e}")
    
    return answer

# --- Inpainting Helpers ---
def make_inpaint_condition(image: Image.Image, image_mask: Image.Image):
    """
    Prepare the control image tensor for inpainting.
    """
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask_np = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    # Mark masked pixels with -1.0
    image_np[image_mask_np > 0.5] = -1.0
    image_np = np.expand_dims(image_np, 0).transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(image_np)
    return image_tensor

# --- Segmentation Helpers ---
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[int]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax']
            )
        )

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(image: Image.Image, labels: List[str], threshold: float = 0.3, detector_id: Optional[str] = None) -> List[DetectionResult]:
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    # Ensure labels end with a period
    labels = [label if label.endswith(".") else label + "." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    detections = [DetectionResult.from_dict(r) for r in results]
    return detections

def segment(image: Image.Image, detection_results: List[DetectionResult], polygon_refinement: bool = False, segmenter_id: Optional[str] = None) -> List[DetectionResult]:
    from transformers import AutoModelForMaskGeneration, AutoProcessor
    device = 0 if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    
    boxes = [det.box.xyxy for det in detection_results]
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(image: Image.Image, labels: List[str], threshold: float = 0.3,
                           polygon_refinement: bool = False, detector_id: Optional[str] = None,
                           segmenter_id: Optional[str] = None) -> Tuple[np.ndarray, List[DetectionResult]]:
    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)
    return np.array(image), detections

def composite_face(person_image: Image.Image, inpainted_image: Image.Image, face_mask: Image.Image) -> Image.Image:
    """
    Composite the original face region from the person image onto the inpainted image.
    """
    original_img = person_image.convert("RGBA")
    inpainted_img = inpainted_image.convert("RGBA")
    face_mask = face_mask.convert("L")
    
    # Extract the face from the original image
    original_face = Image.new("RGBA", original_img.size)
    original_face = Image.composite(original_img, original_face, face_mask)
    
    # Remove face region from inpainted image
    transparent_bg = Image.new("RGBA", inpainted_img.size, (0, 0, 0, 0))
    inpainted_without_face = Image.composite(transparent_bg, inpainted_img, face_mask)
    
    final_img = Image.alpha_composite(inpainted_without_face, original_face)
    return final_img.convert("RGB")

# --- AWS Inference Functions ---

def model_fn(model_dir, hf_token, aws_access_key=None, aws_secret_access_key=None, aws_region='us-west-2', context=None):
    logger.info("========== Entered model_fn ==========")
    
    # Log in to Hugging Face Hub
    from huggingface_hub import login
    login(token=hf_token)
    
    # Load ControlNet and inpainting pipelines
    from diffusers import StableDiffusionControlNetInpaintPipeline, AutoPipelineForInpainting, ControlNetModel
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16)
    inpaint_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        controlnet=controlnet, 
        torch_dtype=torch.float16,
    )
    inpaint_pipe.to('cuda:0')
    
    sdxl_pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    sdxl_pipe.to('cuda:2')
    
    # Load DeepSeek model components
    from transformers import AutoModelForCausalLM
    from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt.to(torch.bfloat16).to("cuda:3").eval()
    
    return {
        "inpaint_pipe": inpaint_pipe,
        "sdxl_pipe": sdxl_pipe,
        "vl_chat_processor": vl_chat_processor,
        "tokenizer": tokenizer,
        "vl_gpt": vl_gpt
    }

def input_fn(serialized_input_data, content_type):
    logger.info("========== Entered input_fn ==========")
    if content_type == "application/json":
        return json.loads(serialized_input_data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    logger.info("========== Entered predict_fn ==========")
    # Expected input keys:
    # "sketch": base64-encoded garment sketch image
    # "person": base64-encoded person image
    # "classification_prompt": (optional) prompt for garment classification
    # "description_prompt": (optional) prompt for detailed description
    # "negative_prompt": (optional) negative prompt for inpainting
    # "num_inference_steps": (optional, default=40)
    # "seed": (optional, default=0)
    # "model_version": (optional) "sd1.5" or "sdxl" (default "sdxl")
    classification_prompt = input_data.get("classification_prompt", "Please classify this garment. Only respond with one phrase.")
    description_prompt = input_data.get("description_prompt", "Give me a caption for this garment. Please describe specific details and attributes of the garment in a photorealistic manner.")
    negative_prompt = input_data.get("negative_prompt", "distorted, face changed, not proportional")
    num_inference_steps = input_data.get("num_inference_steps", 40)
    seed = input_data.get("seed", 0)
    model_version = input_data.get("model_version", "sdxl")
    threshold = input_data.get("segmentation_threshold", 0.3)
    polygon_refinement = input_data.get("polygon_refinement", True)
    detector_id = input_data.get("detector_id", "IDEA-Research/grounding-dino-tiny")
    segmenter_id = input_data.get("segmenter_id", "facebook/sam-vit-base")
    
    # Decode images from base64
    sketch_b64 = input_data.get("sketch", None)
    person_b64 = input_data.get("muse", None)
    if sketch_b64 is None or person_b64 is None:
        raise ValueError("Both 'sketch' and 'person' images must be provided in base64 format.")
    sketch_image = decode_base64_image(sketch_b64)
    person_image = decode_base64_image(person_b64)

    #image_prompt = generate_prompt([bg_image], vl_chat_processor, tokenizer, vl_gpt, prompt=background_description_prompt, max_sequence_length=2048)
    
    # Generate garment classification using DeepSeek
    garment_type = generate_prompt([sketch_image], 
                                   model["vl_chat_processor"], model["tokenizer"], model["vl_gpt"], classification_prompt).strip()
    logger.info(f"Garment classification: {garment_type}")
    
    # Generate a detailed description
    description = generate_prompt([sketch_image], 
                                  model["vl_chat_processor"], model["tokenizer"], model["vl_gpt"], description_prompt).strip()
    logger.info(f"Generated description: {description}")
    
    # Perform segmentation on the person image to obtain garment and face masks
    labels = [garment_type + ".", "face."]
    image_array, detections = grounded_segmentation(person_image, labels, threshold, polygon_refinement, detector_id, segmenter_id)
    if len(detections) < 2:
        raise ValueError("Segmentation did not detect both garment and face.")
    garment_mask = detections[0].mask
    face_mask = detections[1].mask
    garment_mask_image = Image.fromarray(garment_mask).convert("L")
    face_mask_image = Image.fromarray(face_mask).convert("L")
    logger.info("Segmentation completed.")
    
    # Prepare control image tensor for inpainting
    control_image_tensor = make_inpaint_condition(person_image, garment_mask_image)
    
    # Build the inpainting prompt
    inpaint_prompt = description + " Keep the face unchanged. Do not modify the face at all. Only change the specified garment."
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    if model_version == "sd1.5":
        inpaint_pipe = model["inpaint_pipe"]
        output = inpaint_pipe(
            prompt=inpaint_prompt,
            negative_prompt=negative_prompt,
            image=person_image,
            mask_image=garment_mask_image,
            control_image=control_image_tensor,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        inpainted_image = output.images[0]
        logger.info("SD1.5 Inpainting completed.")
    else:
        sdxl_pipe = model["sdxl_pipe"]
        output = sdxl_pipe(
            prompt=inpaint_prompt,
            negative_prompt=negative_prompt,
            image=person_image,
            mask_image=garment_mask_image,
            control_image=control_image_tensor,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        inpainted_image = output.images[0]
        logger.info("SDXL Inpainting completed.")
    
    # Resize inpainted image to match the original person image dimensions
    inpainted_resized = inpainted_image.resize(person_image.size, resample=Image.LANCZOS)
    logger.info("Inpainted image resized.")
    # Apply face correction by compositing the original face from the person image
    final_image = composite_face(person_image, inpainted_resized, face_mask_image)
    logger.info("Face correction applied.")
    return final_image

def output_fn(prediction, accept):
    logger.info("========== Entered output_fn ==========")
    if accept == "application/json":
        image_base64 = encode_image_base64(prediction)
        return {"image": image_base64}
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
