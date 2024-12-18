# Script for generating background image with muse in foreground

# %%

# %%
import os
import requests
from PIL import Image
from io import BytesIO
import replicate
from controlnet_aux import OpenposeDetector


'''
    1st step: generate background image from sketch using finetuned vaporwave model on Replicate
    TODO: try using finetuned model on Runway instead and see if that has better performance
    *Replicate model finetuned on 94 curated vaporwave images*

'''

# %%

REPLICATE_API_TOKEN = '' # replace with Replicate token
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# %%
''''''
def generate_output(input, folder, filename):
    print("reached generate output function")

    output = replicate.run(
        "naishagarwal/vaporwave-model:b7d144ad3297425f0cf99fdf98a9450791592c15755626ad2d41bdeb2fdd45c4",
        input=input
    )   

    for i, image_url in enumerate(output):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        filepath = os.path.join(folder, f"{filename}_{i}.png")
        image.save(filepath)
        print(f"Saved: {filepath}")

# %%
def generate_input(input_image_path, prompt, folder):
    # Default parameter values
    input = {
        "width": 1024,
        "height": 384,
        "input_image": input_image_path,
        "prompt": prompt,
        "refine": "expert_ensemble_refiner",
        "scheduler": 'K_EULER',
        "lora_scale": 0.2,
        "num_outputs": 2,
        "guidance_scale": 4.5,
        "apply_watermark": True,
        "high_noise_frac": 0.8,
        "negative_prompt": "underexposed",
        "prompt_strength": 0.8,
        "num_inference_steps": 50,
        "disable_safety_checker": True
    }

    filename = os.path.splitext(os.path.basename(input_image_path))[0]
    generate_output(input, folder, filename)

# %%
images = ['background-images/forest.png', 'background-images/stage.png']
prompts = ['forest in the style of TOK', 'stage in the style of TOK']
folders = ['predictions-forest-prompt', 'predictions-stage-prompt']

for image_path, prompt, folder in zip(images, prompts, folders):
    os.makedirs(folder, exist_ok=True)
    generate_input(image_path, prompt, folder)


'''
    Second step: Use Control Net to extract muse poses/edges to later add to background image

'''

