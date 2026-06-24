from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch, requests
from io import BytesIO
torch.manual_seed(42) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fn_prompt
def fn_prompt(image, class_name, methods='original', output_file='./demo/prompt.txt'):
    if methods == 'original':
        prompt = f'a photo of {class_name}'
        print("Original class:", class_name)
        with open(output_file, 'w') as f:
            f.write(prompt)
        return prompt
    
    elif methods == 'BLIP-LLM':
        image_caption = get_image_caption(image)
        refined_class = refine_prompt(image_caption, class_name)
        prompt = f'a photo of {refined_class}'
        print("Image caption:", image_caption)
        print("Original class:", class_name)
        print("Refined Prompt:", prompt)
        with open(output_file, 'w') as f:
            f.write(prompt)
        return prompt
    
    else:
        raise NotImplementedError(f"Method '{methods}' is not implemented.")

# with blip (image should use pil module)
def get_image_caption(image_query):
    # load model and processor BLIP
    blip_model_id = "Salesforce/blip-image-captioning-large"
    blip_processor = BlipProcessor.from_pretrained(blip_model_id, use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id) 
    blip_model = blip_model.to(device)
    blip_model.eval()
    inputs = blip_processor(images=image_query, return_tensors="pt").to(device)  # <-- input ke device
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    blip_model = blip_model.to("cpu")
    return caption

# with instruction llm
def refine_prompt(image_caption, class_name):
    # Load model and tokenizer phi3
    llm_model_id = "microsoft/phi-3-mini-128k-instruct"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id, torch_dtype=torch.float16, )
    llm_model = llm_model.to('cpu')
    prompt = f"""
    <|system|>
    You are a smart assistant. Your task is to rewrite the object name based on the given class_name and image_caption.
    Use the image_caption only to clarify or refine the class_name.
    The output should be short (under 100 tokens) and clearly describe the object. Examples:
    - class_name: tablet → <output: digital tablet device> or <output: medicine tablets>
    - class_name: mouse → <output: mouse of animal> or <output: computer mouse>
    - class_name: orange → <output: orange fruits> or <output: orange color>
    - class_name: python → <output: python snake> or <output: python programming language>
    Always base your answer on the class_name.
    <|end|>
    <|user|>
    [image_caption]: {image_caption}
    [class_name]: {class_name}  
    What is the refined object name based on this information?
    <|end|>
    <|assistant|>
    """
    inputs = llm_tokenizer(prompt, return_tensors="pt").to('cpu')  # <-- input ke device
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=100, do_sample=True)
    decoded = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("<|assistant|>\n")[-1].strip()
    refined_class_name = response.split('\n')[-1].strip()
    llm_model = llm_model.to('cpu')
    return refined_class_name.replace('<output: ', '').replace('>', '')

if __name__ == '__main__':
    # input
    # url = 'https://upload.wikimedia.org/wikipedia/commons/3/31/Grapes_and_Banana.jpg'
    # image_path = BytesIO(requests.get(url).content)
    image_path = "./demo_images/Grapes_and_Banana.jpg"
    class_name = "grape"

    # pipeline
    image = Image.open(image_path).convert("RGB")
    prompt = fn_prompt(image, class_name, methods='BLIP-LLM')

    
    
