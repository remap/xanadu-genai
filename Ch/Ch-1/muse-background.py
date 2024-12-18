''' 
    3rd step: insert muse into background using control net image to image (synthetic image for now)

'''

from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline
import torch
from PIL import Image
import requests

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, use_safetensors=True
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

pose_image_path = "muse-images/terpiscore_pose.png"
background_image_path = "predictions-forest-prompt/forest_1.png"
pose_map = Image.open(pose_image_path).resize((512, 512))
background = Image.open(background_image_path).resize((512, 512))

prompt = "The muse Terpsicore in a forest, seamlessly integrated into the background"
synthetic_image = pipe(prompt, image=background, controlnet_conditioning_image=pose_map).images[0]
synthetic_image.save("testing.png")
print("done")