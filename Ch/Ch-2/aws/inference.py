import os
import sys
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append("/opt/ml/InstantID/")
sys.path.append("/opt/ml/InstantID/gradio_demo")

from typing import Tuple

import cv2
import math
import torch
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
# from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download, login

from insightface.app import FaceAnalysis

from style_template import styles
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device, torch_gc
from controlnet_util import openpose #, get_depth_map, get_canny_image
from transparent_background import Remover

import json
import io, base64
import boto3
import gzip
import re

sys.path.append("/opt/ml/model/code/")
sys.path.append("/opt/ml/model/code/utils/")
sys.path.append("/opt/ml/model/code/utils/llm_instructions_folder/")
from llm_instructions import keypoints_prompt, image_description_instruction
from keypoints_utils import extract_keypoints, draw_bodypose
from llm_instructions import tpose_keypoints, DEFAULT_POSES_KEYPOINTS
from body_proportions_utils import rescale_pose_keypoints, rescale_face_box
# DEFAULT_POSE_KEYPOINTS = tpose_keypoints

STYLE_NAMES = list(styles.keys())
# DEFAULT_STYLE_NAME = "Watercolor"
DEFAULT_STYLE_NAME = "(No style)"

def download_s3_folder(bucket_name, s3_prefix, local_dir):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = os.path.relpath(s3_key, s3_prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket_name, s3_key, local_path)

def load_model_from_s3(bucket_name, s3_prefix, local_dir):
    if not os.path.exists(os.path.join(local_dir, "model_index.json")):
        print(f"Downloading model from S3 to {local_dir}...")
        download_s3_folder(bucket_name, s3_prefix, local_dir)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(
    image_pil,
    kps,
    color_list=[
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ],
):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def apply_style(
    style_name: str, positive: str, negative: str = ""
) -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative

from copy import deepcopy
def translate_kps(pose_kps, face_kps_array: np.ndarray, normalize=False, img_size=(1024,1024), translate_other_array=None) -> np.ndarray:
    # define translation as vector between noses
    if isinstance(pose_kps, list):
        nose_from_pose = np.array([pose_kps[0].x, pose_kps[0].y])
        pose_kps_array = np.array([[pose_kps[i].x, pose_kps[i].y] for i in range(len(pose_kps))])
    else: # assume nose keypoints
        nose_from_pose = np.array([pose_kps.x, pose_kps.y])

    normalized_face_kps = deepcopy(face_kps_array)
    array_to_be_translated = deepcopy(pose_kps_array) if translate_other_array is None else deepcopy(translate_other_array)
    if normalize:
        normalized_face_kps[:,0] /= img_size[0]
        normalized_face_kps[:,1] /= img_size[1]
        # array_to_be_translated[:,0] /= img_size[0]
        # array_to_be_translated[:,1] /= img_size[1]

    nose_from_face = normalized_face_kps[2,:]

    translated_kps = np.zeros_like(array_to_be_translated)
    t =  nose_from_face - nose_from_pose
    m,n = translated_kps.shape
    for i in range(m):
        translated_kps[i,:] = array_to_be_translated[i,:] + t

    # if normalize:
    #     translated_kps[:,0] = translated_kps[:,0] * img_size[0]
    #     translated_kps[:,1] = translated_kps[:,1] * img_size[1]

    new_pose_kps = deepcopy(pose_kps)
    for i in range(len(new_pose_kps)):
        new_pose_kps[i].x = translated_kps[i,0]
        new_pose_kps[i].y = translated_kps[i,1]

    return new_pose_kps #translated_kps

class CustomPipeGen():
    def __init__(self, pipe, app, controlnet_identitynet, controlnet_map, controlnet_map_fn, device):
        self.pipe = pipe
        self.app = app
        self.controlnet_identitynet = controlnet_identitynet
        self.controlnet_map = controlnet_map
        self.controlnet_map_fn = controlnet_map_fn
        self.device = device

    def generate_image(self,
        face_image,
        pose_image,
        prompt,
        negative_prompt,
        style_name,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        guidance_scale,
        seed,
        # enable_LCM,
        enhance_face_region,
        pose_kps=None,
        scheduler="DPM++ 2M",
        id_control_guidance_start_end=[0, 1],
        pose_control_guidance_start_end=[0, 1],
        clip_skip=None,
        euler_scheduler=None,
        num_images_per_prompt=1
    ):

        # if enable_LCM:
        #     self.pipe.scheduler = diffusers.LCMScheduler.from_config(self.pipe.scheduler.config)
        #     self.pipe.enable_lora()
        # else:
        #     self.pipe.disable_lora()
        #     scheduler_class_name = scheduler.split("-")[0]

            # add_kwargs = {}
            # if len(scheduler.split("-")) > 1:
            #     add_kwargs["use_karras_sigmas"] = True
            # if len(scheduler.split("-")) > 2:
            #     add_kwargs["algorithm_type"] = "sde-dpmsolver++"
            # scheduler = getattr(diffusers, scheduler_class_name)
            # self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config, **add_kwargs)

        if scheduler == 'Euler':
            if euler_scheduler:
                self.pipe.scheduler = deepcopy(euler_scheduler)
                self.pipe.scheduler.config['timestep_spacing'] = 'trailing'
            else:
                scheduler = 'DPM++ 2M Karras'
        else:
            self.pipe.scheduler.config['timestep_spacing'] = 'leading'

        if scheduler == 'DPM++ 2M':
            self.pipe.scheduler.config['algorithm_type'] = "dpmsolver++"
            self.pipe.scheduler.config['use_karras_sigmas'] = False
        elif scheduler == 'DPM++ 2M SDE':
            self.pipe.scheduler.config['algorithm_type'] = "sde-dpmsolver++"
            self.pipe.scheduler.config['use_karras_sigmas'] = False
        elif scheduler == 'DPM++ 2M Karras':
            self.pipe.scheduler.config['algorithm_type'] = "dpmsolver++"
            self.pipe.scheduler.config['use_karras_sigmas'] = True
        elif scheduler == 'DPM++ 2M SDE Karras':
            self.pipe.scheduler.config['algorithm_type'] = "sde-dpmsolver++"
            self.pipe.scheduler.config['use_karras_sigmas'] = True
        else:
            self.pipe.scheduler.config['algorithm_type'] = "dpmsolver++"
            self.pipe.scheduler.config['use_karras_sigmas'] = False

        if face_image is None:
            logger.error(f"Cannot find any input face image! Please upload the face image")

        if prompt is None:
            prompt = "a person"

        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        # face_image = load_image(face_image_path)
        face_image = resize_img(face_image, max_side=1024)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = self.app.get(face_image_cv2)

        if len(face_info) == 0:
            logger.error(
                f"Unable to detect a face in the image. Please upload a different photo with a clear face."
            )

        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
        img_controlnet = face_image
        if pose_image is not None:
            # pose_image = load_image(pose_image_path)
            pose_image = resize_img(pose_image, max_side=1024)
            img_controlnet = pose_image
            logger.info("POSE IMAGE NOT EMPTY --------------")
            # pose_image_cv2 = convert_from_image_to_cv2(pose_image)
            # face_info = app.get(pose_image_cv2)
            # if len(face_info) == 0:
            #     logger.error(
            #         f"Cannot find any face in the reference image! Please upload another person image"
            #     )
            # face_info = face_info[-1]
            # face_kps = draw_kps(pose_image, face_info["kps"])
            # width, height = face_kps.size

            # translate face keypoints to match the target pose image
            new_keypoints_img = pose_image
            if pose_kps:
                # face_kps_array = translate_kps(
                #     face_info["kps"],
                #     pose_kps,
                #     img_size=(width, height),
                #     normalize=True,
                # )
                # logger.info(f"ORIGINAL FACE KPS: {face_info['kps']}")
                # logger.info(f"NEW FACE KPS: {face_kps_array}")
                face_kps_array = face_info["kps"]
                # logger.info(f"FACE KPS: {face_kps_array}")
                # logger.info(f"POSE KPS: {pose_kps}")
                pose_kps = translate_kps(
                    pose_kps,
                    face_info["kps"],
                    img_size=(width, height),
                    normalize=True,
                )
                # logger.info(f"TRANSLATED POSE KPS: {pose_kps}")
                face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_kps_array)

                # rescale keypoints
                rescaled_keypoints_list, face_keypoints = rescale_pose_keypoints(face_image, face_kps_array, pose_kps)
                new_keypoints_img = draw_bodypose(np.zeros((width, height, 3), dtype=np.uint8), rescaled_keypoints_list)
                new_keypoints_img = Image.fromarray(new_keypoints_img)
                logger.info("New Keypoints image generated successfully.")

                img_controlnet = new_keypoints_img
                old_face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_kps_array)
                face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_keypoints)

        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            # if pose_kps:
            #     bbox_translated_corners = translate_kps(
            #         face_info["kps"],
            #         pose_kps,
            #         img_size=(width, height),
            #         normalize=True,
            #         translate_other_array=np.array([[x1, y1], [x2, y2]])
            #     )
            #     # logger.info(f"BBOX CORNERS {(x1, y1, x2, y2)}")
            #     # logger.info(f"NEW CORNERS: {bbox_translated_corners}")
            #     x1, y1, x2, y2 = bbox_translated_corners[0,0], bbox_translated_corners[0,1], bbox_translated_corners[1,0], bbox_translated_corners[1,1]
            #     # rescale face box
            #     x1, y1, x2, y2 = rescale_face_box((x1, y1, x2, y2), face_info["kps"], face_keypoints)
            #     logger.info(f"New bbox: {(x1, y1, x2, y2)}")

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        control_guidance_start = id_control_guidance_start_end[0]
        control_guidance_end = id_control_guidance_start_end[1]
        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "pose": pose_strength,
                "canny": canny_strength,
                "depth": depth_strength,
            }
            self.pipe.controlnet = MultiControlNetModel(
                [self.controlnet_identitynet]
                + [self.controlnet_map[s] for s in controlnet_selection]
            )
            try:
                logger.info(f"Controlnet device: {self.pipe.controlnet.device}")
            except:
                logger.info(f"Controlnet devices: {self.pipe.controlnet[0].device}, {self.pipe.controlnet[1].device}")

            control_scales = [float(identitynet_strength_ratio)] + [
                controlnet_scales[s] for s in controlnet_selection
            ]
            control_images = [face_kps] + [img_controlnet.resize((width, height))]
            #     self.controlnet_map_fn[s](img_controlnet).resize((width, height))
            #     for s in controlnet_selection
            # ]
            # if pose controlnet, test out a direct pose image
            logger.info(f"===== CONTROLNET SELECTION:  {controlnet_selection}")
            if 'pose' in controlnet_selection:
                logger.info("ENTERED CONTROLNET SELECTION")
                # print(len(control_images))
                control_images[1] = img_controlnet #pose_image
                # print(len(control_images))
                control_guidance_start = [id_control_guidance_start_end[0], pose_control_guidance_start_end[0]]
                control_guidance_end = [id_control_guidance_start_end[1], pose_control_guidance_start_end[1]]
        else:
            self.pipe.controlnet = self.controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info("Starting image generation...")
        logger.info(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

        self.pipe.set_ip_adapter_scale(adapter_strength_ratio)
        
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        torch.cuda.empty_cache()

        return images[0], face_kps, old_face_kps, new_keypoints_img


# Load the model and pipeline
def model_fn(model_dir, hf_token=None, aws_region='us-west-2', context=None):
    # Set SDK socket timeout to 180 seconds
    import botocore.config
    config = botocore.config.Config(
        read_timeout=30,
        connect_timeout=30,
        retries={'max_attempts': 10}
    )

    logger.info("========== Entered model_fn ==========")
    logger.info(f"Diffusers version: {diffusers.__version__}")

    device = "cuda:0"
    dtype = torch.float16
    # STYLE_NAMES = list(styles.keys())
    # DEFAULT_STYLE_NAME = "Watercolor"

    base_dir = "/opt/ml/InstantID"
    bucket_name = "sagemaker-generative-models-for-deployment"
    s3 = boto3.client("s3")

    # download models
    logger.info("Downloading IdentityNet...")
    # hf_hub_download(
    #     repo_id="InstantX/InstantID",
    #     filename="ControlNetModel/config.json",
    #     local_dir= base_dir+"/checkpoints",
    # )
    # hf_hub_download(
    #     repo_id="InstantX/InstantID",
    #     filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    #     local_dir=base_dir+"/checkpoints",
    # )
    download_s3_folder(
        bucket_name=bucket_name,
        s3_prefix="instantid/ControlNetModel/",
        local_dir=base_dir+"/checkpoints/ControlNetModel",
    )
    logger.info("Downloading IP-Adapter...")
    # hf_hub_download(
    #     repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir=base_dir+"/checkpoints"
    # )
    s3.download_file(
        Bucket=bucket_name,
        Key="instantid/ip-adapter.bin",
        Filename=base_dir+"/checkpoints/ip-adapter.bin",
    )

    # Load face encoder
    face_app = FaceAnalysis(
        name="antelopev2",
        root=base_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f"{base_dir}/checkpoints/ip-adapter.bin"
    controlnet_path = f"{base_dir}/checkpoints/ControlNetModel"

    # Load pipeline face ControlNetModel
    logger.info("Loading ControlNetModel...")
    controlnet_identitynet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=dtype
        )

    # controlnet-pose
    # controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
    controlnet_pose_model = base_dir+"/checkpoints/PoseNet"
    download_s3_folder(
        bucket_name=bucket_name,
        s3_prefix="instantid/PoseNet/",
        local_dir=controlnet_pose_model,
    )
    # controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
    # controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

    logger.info("Loading Pose ControlNet...")
    controlnet_pose = ControlNetModel.from_pretrained(
        controlnet_pose_model, torch_dtype=dtype, use_safetensors=False
    ).to(device)
    # controlnet_canny = ControlNetModel.from_pretrained(
    #     controlnet_canny_model, torch_dtype=dtype
    # ).to(device)
    # controlnet_depth = ControlNetModel.from_pretrained(
    #     controlnet_depth_model, torch_dtype=dtype
    # ).to(device)

    controlnet_map = {
        "pose": controlnet_pose,
        # "canny": controlnet_canny,
        # "depth": controlnet_depth,
    }
    controlnet_map_fn = {
        "pose": openpose,
        # "canny": get_canny_image,
        # "depth": get_depth_map,
    }

    logger.info("Loading StableDiffusionXLInstantIDPipeline...")
    if 0: #hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("Loading Juggernaut-XI-v11")
        pretrained_model_name_or_path="RunDiffusion/Juggernaut-XI-v11"
    else:
        logger.info("Loading YamerMIX_v8")
        # pretrained_model_name_or_path="wangqixun/YamerMIX_v8"
        pretrained_model_name_or_path = base_dir+"/checkpoints/YamerMIX_v8"
        load_model_from_s3(
            bucket_name=bucket_name,
            s3_prefix="instantid/YamerMIX_v8/",
            local_dir=pretrained_model_name_or_path
        )

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            pretrained_model_name_or_path,
            controlnet=[controlnet_identitynet],
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

    original_Euler_scheduler = deepcopy(diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config))

    pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    pipe.load_ip_adapter_instantid(face_adapter)
    # Initialize custom pipe
    logger.info("Instantiating CustomPipeGen")
    custom_pipe = CustomPipeGen(pipe, 
                                face_app, 
                                controlnet_identitynet, 
                                controlnet_map, 
                                controlnet_map_fn, 
                                device)

    # Instantiate Bg Remover
    logger.info("Instantiating Bg Remover")
    bg_remover = Remover(device=device)

    # Load Bedrock
    logger.info("Loading Bedrock runtime...")
    bedrock_runtime = boto3.client('bedrock-runtime', config=config, region_name=aws_region)

    return {
        'pipe': pipe,
        'custom_pipe': custom_pipe,
        'face_app': face_app,
        "controlnet_identitynet": controlnet_identitynet,
        'controlnet_map': controlnet_map,
        'controlnet_map_fn': controlnet_map_fn,
        'bedrock_runtime': bedrock_runtime,
        'bg_remover': bg_remover,
        'original_Euler_scheduler': original_Euler_scheduler
    }

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
    num_inference_steps = input_data.get("num_inference_steps", 35)
    seed = min(input_data.get("seed", 637678054), np.iinfo(np.int32).max)
    style_name = input_data.get("style_name", DEFAULT_STYLE_NAME)

    face_image_base64 = input_data["face_image"]
    pose_image_base64 = input_data.get("pose_sketch", None)
    garment_image_base64 = input_data["garment_sketch"]

    image_prompt = input_data.get("image_prompt", None)
    if image_prompt:
        if len(image_prompt) <= 1:
            image_prompt = None
    llm_prompt = input_data.get("llm_prompt", None)
    if llm_prompt:
        if len(llm_prompt) <= 1:
            llm_prompt = None

    identitynet_strength_ratio = input_data.get("identitynet_strength_ratio", 0.95)
    adapter_strength_ratio = input_data.get("adapter_strength_ratio", 0.95)
    pose_strength = input_data.get("pose_strength", 0.75)
    canny_strength = input_data.get("canny_strength", 0.35)
    depth_strength = input_data.get("depth_strength", 0.35)
    controlnet_selection = input_data.get("controlnet_selection", ["pose"])
    guidance_scale = input_data.get("guidance_scale", 3)
    enhance_face_region = input_data.get("enhance_face_region", True)
    scheduler = input_data.get("scheduler", "DPM++ 2M")

    default_negative_prompt = "underdressed, nudity, nsfw, extra fingers, mutated hands, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs"
    negative_prompt = input_data.get("negative_prompt", default_negative_prompt)

    id_control_guidance_start_end = input_data.get("id_control_guidance_start_end", [0.0, 1.0])
    pose_control_guidance_start_end = input_data.get("pose_control_guidance_start_end", [0.0, 1.0])
    clip_skip = input_data.get("clip_skip", 0)

    num_images_per_prompt = input_data.get("num_images_per_prompt", 1)

    # Convert base64 to image
    if face_image_base64 is not None:
        # Decode image from Base64
        face_image_data = base64.b64decode(face_image_base64)
        face_image = Image.open(io.BytesIO(face_image_data)).convert("RGB")
        width, height = face_image.size
    else:
        logger.error("No face image provided")
        raise ValueError("Input must include a face image ")
    
    if pose_image_base64 is not None:
        # Decode image from Base64
        pose_image_data = base64.b64decode(pose_image_base64)
        # Convert base64 to tensor if needed
        # image = torch.tensor(image)  # Example
        pose_image = Image.open(io.BytesIO(pose_image_data)).convert("RGB")
    else:
        logger.info("No pose image provided, defaulting to T-pose.")

    if garment_image_base64 is not None:
        # Decode image from Base64
        garment_image_data = base64.b64decode(garment_image_base64)
        garment_image = Image.open(io.BytesIO(garment_image_data)).convert("RGB")
    else:
        logger.info("No garment image provided.")
        garment_image = None

    #  Parse model
    bedrock_runtime = model["bedrock_runtime"]

    # Generate image prompt ----------------------------------------
    if image_prompt is not None:
        logger.info("Image prompt provided, bypassing call to LLM.")
        prompt = image_prompt
    else:
        logger.info("No image prompt provided, calling LLM to generate prompt.")
        # person_garment_prompt_for_llm = "Attached is a sketch of a garment. "+\
        #                             "Describe what it represents in a manner that constitutes a good prompt " +\
        #                             "for a text-to-image model to generate a realistic photo of the clothing piece(s). "+\
        #                             "Alongside the garment sketch, there is also the picture of a person. "+\
        #                             "Depending on the person's photo, generate a good prompt for a text-to-image "+\
        #                             "that combines the person and the garment in the following way: "+\
        #                             "A photo of a {man/woman} wearing {detailed attire description highlighting the generated garment interpretation, including color and any motif, pattern or texture}. "+\
        #                             "Only output this text without any other wrapper text or information."
        
        person_garment_prompt_for_llm = "Attached is a stylized rendition of a clothing fabric. Describe the style of this artistic rendering, especially the color palette and the overall style and effect."+\
                                    "Alongside the stylized rendition of a clothing fabric, there is also the picture of a person. "+\
                                    "Depending on the person's photo, generate a good prompt for a text-to-image diffusion model"+\
                                    "describing the person wearing a garment suitable for a Xanadu reproduction but closely following the color palette and style of the fabric rendition in the following way: "+\
                                    "A photo of a {man/woman} wearing {new detailed attire description, including the above color palette and any suitable motif, pattern or texture}. Do NOT refer to the fabric at any point in the prompt."+\
                                    "Only output this text without any other wrapper text or information."
        
        if llm_prompt is not None:
            logger.info("LLM prompt provided.")
            # llm_text_payload = llm_prompt
            person_garment_prompt_for_llm = llm_prompt
        else:
            logger.info("No LLM prompt provided, using default prompt.")
        # Generate prompt for image of person + garment
        if garment_image is not None:
            logger.info("Garment image provided. Generating prompt for image of person + garment.")
            
            garment_payload = generate_payload_for_llm(person_garment_prompt_for_llm, [garment_image_base64, face_image_base64])
            response = bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(garment_payload),
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            generated_garment_person_prompt = response_body['content'][0]['text']
            logger.info(f"LLM Generated garment-person prompt: {generated_garment_person_prompt}")

            prompt = generated_garment_person_prompt

        else:
            logger.info("Garment image not provided. Defaulting to prompt for image of person.")
            prompt = "A photo of a person."

    logger.info(f"Prompt for image generation: {prompt}")
    
    # Generate Pose --------------------------------------
    llm_text_payload = f"This is a sketch representing "+\
            f"a person in a particular pose, which may or may not be clear. "\
            f"If it is not, do your best interpretation. "+\
            f"I want you to follow the instructions below. \n"+\
            f"{image_description_instruction}\n If some parts are occluded or not visible, take a guess nonetheless (do not say 'not clearly defined' or something similar). \n"+\
            f"If you cannot interpret the sketch or you don't find a person or can't identify the pose, "+\
            f"then please return *exactly* the following text and *nothing else*: 'No person or pose detected'."

    # Select default pose
    DEFAULT_POSE_NAME, DEFAULT_POSE_KEYPOINTS = random.choice(list(DEFAULT_POSES_KEYPOINTS.items()))
    logger.info(f"Default pose name: {DEFAULT_POSE_NAME}")
    if pose_image is not None:
        logger.info("Pose image provided, calling LLM to generate pose text.")
        # Invoke the model for pose text
        pose_payload = generate_payload_for_llm(llm_text_payload, [pose_image_base64])
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(pose_payload),
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        generated_pose_text = response_body['content'][0]['text']
        generated_pose_text = re.sub(r"^```(?:markdown)?\n|\n```$", "", generated_pose_text.strip())
        logger.info(f"LLM Generated text for pose: {generated_pose_text}")

        if "no person or pose detected" in generated_pose_text.lower():
            logger.info("No person or pose detected in pose image. Falling back to default pose.")
            # generated_keyposes_text = DEFAULT_POSE_KEYPOINTS
            keypoints_json_text = DEFAULT_POSE_KEYPOINTS

        else:

            # Generate keypoints text
            logger.info("Generating keypoints text.")
            llm_keypoints_prompt = keypoints_prompt + f"\n Now here's the structured human pose description "+ \
                    f"for which I want you to generate the corresponding 2D keypoints normalized to [0,1]:\n"+\
                    f"{generated_pose_text}\n"+"Do not forget that smaller y values mean higher in the image. \n Please return only the raw keypoints JSON string! Do not include any other text."

            keypoints_payload = generate_payload_for_llm(llm_keypoints_prompt)
            response = bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(keypoints_payload),
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            generated_keyposes_text = response_body['content'][0]['text']
            logger.info(f"LLM Generated Keypoints text: {generated_keyposes_text}")

            keypoints_json_text = extract_keypoints_json_text(generated_keyposes_text)
            if keypoints_json_text is None:
                logger.info("No keypoints detected in pose image. Falling back to default pose.")
                # generated_keyposes_text = DEFAULT_POSE_KEYPOINTS
                keypoints_json_text = DEFAULT_POSE_KEYPOINTS

    else:
        logger.info("No pose image provided, cannot generate prompt. Falling back to default pose.")
        # generated_keyposes_text = DEFAULT_POSE_KEYPOINTS
        keypoints_json_text = DEFAULT_POSE_KEYPOINTS

    # Generate keypoints image
    logger.info("Generating keypoints image.")
    actual_keypoints_list = extract_keypoints(keypoints_json_text) #generated_keyposes_text)
    # check that nose_to_notch segment is not 0
    if (actual_keypoints_list[1] - actual_keypoints_list[0]).norm() <= 0.01:
        logger.info("Nose-to-notch segment is 0. Falling back to default pose.")
        # generated_keyposes_text = DEFAULT_POSE_KEYPOINTS
        keypoints_json_text = DEFAULT_POSE_KEYPOINTS
        actual_keypoints_list = extract_keypoints(keypoints_json_text)
    keypoints_img = draw_bodypose(np.zeros((width, height, 3), dtype=np.uint8), actual_keypoints_list)
    keypoints_img = Image.fromarray(keypoints_img)
    logger.info("Keypoints image generated successfully.")

    # Generate image ============================================================
    custom_pipe = model["custom_pipe"]
    original_Euler_scheduler = model['original_Euler_scheduler']
    
    # negative_prompt = "nudity, nsfw"
    # negative_prompt = "nudity, nsfw, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs"
    # negative_prompt = "nudity, nsfw, extra fingers, mutated hands, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs"

    json_for_logger = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "style_name": style_name,
        "num_inference_steps": num_inference_steps,
        "identitynet_strength_ratio": identitynet_strength_ratio,
        "adapter_strength_ratio": adapter_strength_ratio,
        "pose_strength": pose_strength,
        "controlnet_selection": controlnet_selection,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "enhance_face_region": enhance_face_region
    }
    logger.info(f"Generating image with these params: {json.dumps(json_for_logger)}")
    gen_image, face_kps, old_face_kps, new_keypoints_img = custom_pipe.generate_image(
                                                                            face_image,
                                                                            keypoints_img,
                                                                            prompt,
                                                                            negative_prompt,
                                                                            style_name,
                                                                            num_inference_steps,
                                                                            identitynet_strength_ratio,
                                                                            adapter_strength_ratio,
                                                                            pose_strength,
                                                                            canny_strength,
                                                                            depth_strength,
                                                                            controlnet_selection,
                                                                            guidance_scale,
                                                                            seed,
                                                                            # enable_LCM,
                                                                            enhance_face_region,
                                                                            pose_kps=actual_keypoints_list,
                                                                            scheduler=scheduler,
                                                                            id_control_guidance_start_end=id_control_guidance_start_end,
                                                                            pose_control_guidance_start_end=pose_control_guidance_start_end,
                                                                            clip_skip=clip_skip,
                                                                            euler_scheduler=original_Euler_scheduler,
                                                                            num_images_per_prompt=num_images_per_prompt
                                                                        )
    
    # encode image
    logger.info("Image generated successfully! \n Encoding generated image.")
    encoded_img = encode_image_base64(gen_image)

    # Removing background
    logger.info("Removing background from generated image.")
    bg_remover = model["bg_remover"]
    no_bg_image = remove_background(gen_image.convert("RGBA"), bg_remover)
    logger.info("Background removed from generated image.")

    # output
    output = {
                "Image_prompt": prompt,
                "Image": encoded_img,
                "Keypoints_image": encode_image_base64(new_keypoints_img),
                "Face_keypoints": encode_image_base64(face_kps),
                "No_background_image": encode_image_base64(no_bg_image)
            }

    return output 

def remove_background(
        image: Image,
        bg_remover: Remover = None,
        force: bool = False,
        **transparent_background_kwargs,
    ) -> Image:
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            do_remove = False
        do_remove = do_remove or force
        if do_remove:
            image = bg_remover.process(
                image.convert("RGB"), **transparent_background_kwargs
            )
        return image

# generate payload for bedrock
def generate_payload_for_llm(payload_text_for_llm, images_for_llm=None):
    logger.info("Generating payload for LLM")
    llm_content = [{"type": "text", "text": payload_text_for_llm}]
    if images_for_llm:
        for img in images_for_llm:
            llm_content.append({
                "type": "image", 
                "source": {
                    "type": "base64", 
                    "media_type": "image/jpeg", 
                    "data": img
                    }
                })
    # payload_with_image =
    return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999,
            "messages": [
                {
                    "role": "user",
                    "content": llm_content
                }
            ]
        }

# def extract_keypoints_json_text(text):
#     # Match a JSON block that starts with "keypoints":
#     pattern = r'({\s*"keypoints"\s*:\s*{.*?}})'  # non-greedy to avoid overmatch
#     match = re.search(pattern, text, re.DOTALL)

#     if match:
#         try:
#             # return json.loads(match.group(1))
#             return match.group(1)
#         except json.JSONDecodeError as e:
#             logger.info("JSON decode error:", e)
#             return None
#     else:
#         logger.info("No match found.")
#         return None

def extract_keypoints_json_text(text):
    # 1. Remove Markdown block formatting if present
    text = re.sub(r"^```(?:json)?\n|\n```$", "", text.strip())

    # 2. Extract a JSON-like dict that starts with "keypoints"
    # Match JSON dict even if it's inside a longer string
    json_candidate = None

    # Try matching JSON-like structure with double quotes
    # match = re.search(r'({\s*"keypoints"\s*:\s*{.*?}})', text, re.DOTALL)
    # match = re.search(r'({\s*"keypoints"\s*:\s*{.*?}\s*})', text, re.DOTALL)
    match = re.search(r'({\s*"keypoints"\s*:\s*{.*?}\s*}\s*})', text, re.DOTALL)
    if match:
        json_candidate = match.group(1)
        logger.info("Matched JSON-like structure with double quotes.")
        # logger.info(json_candidate)
        return json_candidate

    # Try matching single-quoted Python dict as fallback
    if not json_candidate:
        # match = re.search(r"({\s*'keypoints'\s*:\s*{.*?}})", text, re.DOTALL)
        match = re.search(r"({\s*'keypoints'\s*:\s*{.*?}\s*}\s*})", text, re.DOTALL)
        if match:
            json_candidate = match.group(1).replace("'", '"')
            logger.info("Matched JSON-like structure with single quotes.")
            # logger.info(json_candidate)
            return json_candidate

    # Try matching escaped JSON string (escaped double quotes)
    if not json_candidate:
        # match = re.search(r'({\\?"keypoints\\?":\\?{.*?}\\?})', text)
        # match = re.search(r'({\\?"keypoints\\?":\\?{.*?}\\?}\s*})', text)
        match = re.search(r'({\\?"keypoints\\?"\s*:\s*\\?{.*?}\s*\\?}\s*})', text)
        if match:
            logger.info("Matched escaped JSON string.")
            try:
                # Unescape and parse
                unescaped = bytes(match.group(1), "utf-8").decode("unicode_escape")
                # return json.loads(unescaped)
                return unescaped
            except Exception as e:
                print("Error parsing escaped JSON:", e)
                return None

    if not json_candidate:
        logger.info("No valid JSON candidate found.")
        return None

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