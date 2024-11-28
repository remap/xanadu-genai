# finetuning stable diffusion model with vaporwave images
# inference on calliope model 

from diffusers import StableDiffusionPipeline

model_path = "./calliope-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe = pipe.to("cuda")  # If using GPU

prompt = "a photo of sks Calliope on a boat"
image = pipe(prompt).images[0]
#image.show()

image.save("calliope_boat_image.png")  # Save as PNG
print("Image saved as 'generated_image.png'")


