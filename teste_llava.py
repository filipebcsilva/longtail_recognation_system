import ollama
import os
import glob
import pandas as pd
from datetime import timedelta
import time


PASTA_IMAGENS = "./bdd100"
ARQUIVO_SAIDA = "resultado_bdd100.csv"
MODELO = "llava:7b"


PERGUNTAS = ["Number of people","Number of cars","Is it daytime or nighttime","Is there any traffic light","What is the weather conditions","Number of signs","Number of childrens"]

def perguntar_ao_llava(caminho_imagem, pergunta):
    prompt = f"""
    Look at the image and answer the question.
    Question: {pergunta}
    
    RULES:
    - Answer ONLY with the result. No sentences.
    - If it's a count question, return ONLY the integer (e.g., 3).
    - If it's yes/no, return ONLY "yes" or "no".
    - If it's text,keep it concise and lowercase (e.g., 'blue', 'metal').",
    """
    
    try:
        response = ollama.chat(
            model=MODELO,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [caminho_imagem]
            }],
            options={"temperature": 0.0}
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Erro na api: {e}")
        return "error"

def atualizar_csv(dados_da_imagem):
    nome_arquivo = dados_da_imagem['Image_Name']
    
    df_novo = pd.DataFrame([dados_da_imagem])
    
    if os.path.exists(ARQUIVO_SAIDA) and os.path.getsize(ARQUIVO_SAIDA) > 0:
        try:
            df_antigo = pd.read_csv(ARQUIVO_SAIDA)
            
            df_antigo = df_antigo[df_antigo['Image_Name'] != nome_arquivo]
            
            df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
            
        except pd.errors.EmptyDataError:
            print(" Arquivo CSV estava corrompido ou vazio. Recriando...")
            df_final = df_novo
    else:
        df_final = df_novo
    
    cols = ['Image_Name'] + [c for c in df_final.columns if c != 'Image_Name']
    df_final = df_final[cols]
    
    df_final.to_csv(ARQUIVO_SAIDA, index=False)

def main():
    extensoes = ("*.jpg", "*.jpeg", "*.png")
    imagens = []
    for ext in extensoes:
        imagens.extend(glob.glob(os.path.join(PASTA_IMAGENS, ext)))
    
    imagens = sorted(list(set(imagens)))
    
    inicio = time.perf_counter()
    
    for img_path in imagens:
        nome_arquivo = os.path.basename(img_path)
        
        dados_da_imagem = {'Image_Name': nome_arquivo}
        
        for pergunta in PERGUNTAS:
            
            resposta = perguntar_ao_llava(img_path, pergunta)
            
            resposta = resposta.replace(".", "")
            
            dados_da_imagem[pergunta] = resposta
            
        atualizar_csv(dados_da_imagem)

    fim = time.perf_counter()

    tempo_execucao = fim - inicio

    print(f"O código levou {tempo_execucao:.4f} segundos para rodar.")
if __name__ == "__main__":
    if not os.path.exists(PASTA_IMAGENS):
        print(f"A pasta '{PASTA_IMAGENS}' não existe.")
    else:
        main()