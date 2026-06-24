from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16,
).to(device)

# function for generate number of images base on given prompt
def generate_image_from_prompt(prompt, num_image=5, save=True, img_id_ext='inference.jpg'):
    output_dir = "./demo/sd_exemplars"
    os.makedirs(output_dir, exist_ok=True)
    images = pipe(prompt, num_images_per_prompt=num_image).images
    if save:
        for i, img in enumerate(images):
            file_name = prompt.replace(" ", "_") + f"_{i}_" + img_id_ext
            save_path = os.path.join(output_dir, file_name).replace("\n", "")
            img.save(save_path)
    pipe.to('cpu')
    return images


if __name__ == '__main__':
    generate_image_from_prompt('fresh cut', num_image=5, save=True, img_id_ext='testing.jpg')