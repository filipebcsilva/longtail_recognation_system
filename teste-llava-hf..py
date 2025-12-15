import torch  
import cv2 
import json 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
from transformers import pipeline  
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


# pipeline = pipeline(  
#     task="image-text-to-text",  
#     model="llava-hf/llava-v1.6-mistral-7b-hf",  
#     device=0,  
#     dtype=torch.bfloat16  
# )  


PERGUNTAS = [
    "How many people are there?", 
    "How many cars are there?", 
    "How many buses are there?", 
    "How many trucks are there?",
    "Is there any RED traffic light?", 
    "Is there any GREEN traffic light?", 
    "Is there any YELLOW traffic light?",
    "How many bikes are there?",
    "How many motorcycles are there?",
    "Are there any traffic signs?",
    "How many riders in motorcyles or bikes are there?",
    "How many trains (those that move over railroads) are there?",
    "What is the weather among the following options 'mostly clear' / 'rainy' / 'snowy' / 'foggy' or 'cloudy'? Answer 'undefined' if it is not possible to infer from the image.",
    "What is the time of the day among the following options 'daytime', 'night' or 'dusk/dawn' ? Answer 'undefined' if it is not possible to infer from the image.",
    "What is the scene among the following options 'city streets', 'highway', 'residential', 'parking lot', 'gas station', 'tunnel'? Answer 'undefined' if it is not possible to infer from the image.",
]

IMGS_DIR = "./bdd100k-dataset/bdd100k/bdd100k/images/100k/train/trainA/"
img = Image.open(f"{IMGS_DIR}/00091078-59817bb0.jpg")
#img = img.resize((400, 400), Image.Resampling.LANCZOS)

quant_config = BitsAndBytesConfig(  
    load_in_4bit=True,  
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4"  
)

plt.imshow(img)


processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quant_config, device_map="auto")


for question in PERGUNTAS:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"You are an excelent program for analysing images. Please, look into this image and answer the following question: '{question}'. Answer only the question and nothing else. If your answer is numberic, answer with the number. If it is categorical, answer only the category name. If boolean, answer True or False."},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    inputs = processor(img, prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=100)

    print(question)
    print(processor.decode(output[0], skip_special_tokens=True))
    print('--------------------')

plt.show()