import os
import logging

# 1. Configura logs DETALHADOS para ver o que o HuggingFace está fazendo escondido
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("transformers")
logger.setLevel(logging.DEBUG)

print("--- [1] Importando bibliotecas... ---")
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    print("--- [2] Bibliotecas importadas. ---")
except Exception as e:
    print(f"ERRO NA IMPORTAÇÃO: {e}")
    exit()

# Força CPU para isolar problema de driver
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_id = "openai-community/gpt2" # Modelo pequeno e seguro

print(f"--- [3] Tentando baixar SOMENTE o Tokenizer do {model_id}... ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, force_download=True)
    print("--- [4] Tokenizer baixado com sucesso! ---")
except Exception as e:
    print(f"ERRO NO TOKENIZER (INTERNET/CACHE): {e}")
    exit()

print(f"--- [5] Tentando baixar o MODELO (pode demorar)... ---")
try:
    model = AutoModelForCausalLM.from_pretrained(model_id, force_download=True)
    print("--- [6] Modelo baixado! ---")
except Exception as e:
    print(f"ERRO NO MODELO: {e}")
    exit()

print("--- [7] Montando o pipeline... ---")
try:
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    print("--- [8] SUCESSO TOTAL! O pipeline foi criado. ---")
except Exception as e:
    print(f"ERRO NO PIPELINE: {e}")