import os
import time
import logging
import glob
import torch
import transformers
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO)
transformers.utils.logging.set_verbosity_info()

print("Carregando modelo e processador...")
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct", 
    dtype= torch.bfloat16, 
    device_map="cuda",
    trust_remote_code=True,
    resume_download=True,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct",trust_remote_code=True)
print("Modelo carregado!\n")

print("="*30)
print(f"CUDA disponível? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória VRAM alocada: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Memória VRAM reservada: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

print(f"Dispositivo onde o modelo está: {model.device}")
print("="*30)

def perguntar_ao_modelo(image_path, question_text):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=False,  
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_beams=1
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    except Exception as e:
        print(f"Erro na inferência: {e}")
        return "ERRO"

imagens = []
imagens.extend(glob.glob(os.path.join("./teste_bdd","*.jpg")))

imagens = sorted(list(set(imagens)))
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
    "What is the time of the day among the following options 'daytime'/ 'night' or 'dusk/dawn' ? Answer 'undefined' if it is not possible to infer from the image.",
    "What is the scene among the following options 'city streets'/ 'highway'/ 'residential'/ 'parking lot'/ 'gas station'/'tunnel'? Answer 'undefined' if it is not possible to infer from the image.",
]

total_imgs = len(imagens)
buffer_dados = [] 
print(total_imgs)
inicio = time.perf_counter()

for i, img_path in enumerate(imagens):
    nome_arquivo = os.path.basename(img_path)
    print(f"[{i+1}/{total_imgs}] Processando: {nome_arquivo}")
    
    dados_da_imagem = {'Image_Name': nome_arquivo}

    for pergunta in PERGUNTAS:
        resposta = perguntar_ao_modelo(img_path, pergunta)
        
        dados_da_imagem[pergunta] = resposta
        
    buffer_dados.append(dados_da_imagem)

fim = time.perf_counter()
tempo_execucao = fim - inicio

print(f"\nTempo total: {tempo_execucao:.4f}s")
print("-" * 30)
print("DADOS FINAIS COLETADOS:")

from pprint import pprint
pprint(buffer_dados)