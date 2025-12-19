import os
import time
import glob
import pandas as pd
import ollama
from ollama import Client

# --- CONFIGURAÇÕES ---
PASTA_IMAGENS = "./bdd_3k"
ARQUIVO_SAIDA = "bdd7k_llava_34b_q8.csv"
MODELO = "llava:34b-v1.6-q8_0" 
INTERVALO_CHECKPOINT = 50 
CHECKPOINT = True  

client_gpu2 = Client(host='http://127.0.0.1:11435')

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
    "What is the weather among the following options 'mostly clear' / 'rainy' / 'snowy' / 'foggy' / 'cloudy' or 'undefined'? ",
    "What is the time of the day among the following options 'daytime'/ 'night' or 'dusk/dawn' or 'undefined'?",
    "What is the scene among the following options 'city streets' / 'highway / 'residential' /  'parking lot' / 'gas station' / 'tunnel' or 'undefined'?",
]
# PERGUNTAS = [
#     "Is there any artificial lights?",
#     "Is there any shadow?",
#     "Is there any twilight (Morning or evening)?",
#     "Are there any animal on the road?",
#     "Is there any cone or other objects on the road?",
#     "Is there any work on the road?",
#     "Is there any scholar bus?",
#     "Is there any children on the road?",
#     "What is the weather among the following options 'mostly clear' / 'rainy' / 'snowy' / 'foggy' / 'cloudy' or 'undefined'? ",
#     "What is the time of the day among the following options 'daytime'/ 'night' or 'dusk/dawn' or 'undefined'?",
#     "Are there any people in unusual positions (fallen on the ground)?",
#     "Are there any cyclists making dangerous moves?",
#     "Are there any pedestrians making dangerous moves?",
#     "Are there any vehicle crossing a red or yellow light?",
#     "Are there any poor lighting conditions (e.g., glare, shadows) affecting visibility?",
#     "Are there any vehicles (mainly trucks or buses) obstructing visibility?"
# ]
def limpar_resposta(texto):
    if not texto:
        return "undefined"
    
    texto_limpo = texto.replace('"', '').replace("'", "").replace(".", "").replace(",", "")
    texto_limpo = texto_limpo.strip() 
    
    if texto_limpo.lower() == 'yes':
        return 'Yes'
    if texto_limpo.lower() == 'no':
        return 'No'
        
    return texto_limpo

def perguntar_ao_llava(caminho_imagem, pergunta):
    prompt = f"""
    Look at the image and answer the question: "{pergunta}"
    
    CRITICAL RULES:
    1. Output ONLY the answer. Do NOT use full sentences.
    2. Do NOT use punctuation (no periods, no commas, no quotes).
    3. If numeric, output ONLY the digit (e.g., 3).
    4. If boolean, output ONLY 'Yes' or 'No'.
    5. If categorical, output ONLY the category name.
    """
    print(f"Perguntando: {pergunta}")
    try:
        response = client_gpu2.generate(
            model=MODELO,
            prompt=prompt,
            images=[caminho_imagem],
            options={
                "temperature": 0.0, 
                "num_ctx": 4096,
            },
        )
        
        texto_cru = response['response']
        print(f"Respondi essa pergunta {pergunta}")
        return limpar_resposta(texto_cru)
        
    except Exception as e:
        print(f"Erro na api: {e}")
        return "error"

def salvar_batch_rapido(lista_novos_dados):
    if not lista_novos_dados:
        return

    print(f"Salvando Checkpoint ({len(lista_novos_dados)} imagens)...")

    df_novo_batch = pd.DataFrame(lista_novos_dados)

    cols_ordenadas = ['Image_Name'] + PERGUNTAS
    
    for col in cols_ordenadas:
        if col not in df_novo_batch.columns:
            df_novo_batch[col] = "undefined"

    df_novo_batch = df_novo_batch.reindex(columns=cols_ordenadas)
    
    arquivo_existe = os.path.exists(ARQUIVO_SAIDA) and os.path.getsize(ARQUIVO_SAIDA) > 0
    escrever_cabecalho = not arquivo_existe
    
    df_novo_batch.to_csv(ARQUIVO_SAIDA, mode='a', header=escrever_cabecalho, index=False)
    
    with open("last_image2.txt", 'w') as f:
        f.write(lista_novos_dados[-1]['Image_Name'])

def main():
    imagens = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imagens.extend(glob.glob(os.path.join(PASTA_IMAGENS, ext)))
    
    imagens = sorted(list(set(imagens)))
    total_imgs = len(imagens)
    if CHECKPOINT:
        last_image = 0
        for i ,img in enumerate(imagens):
            nome_arquivo = os.path.basename(img)
            with open("last_image2.txt", 'r') as f:
                last_processed = f.read().strip()
            if nome_arquivo == last_processed:
                last_image = i
                break
        imagens = imagens[last_image +1 :]
    total_imgs = len(imagens)
    print(f"Iniciando processamento de {total_imgs} imagens.")

    inicio = time.perf_counter()
    
    buffer_dados = []
     
    for i, img_path in enumerate(imagens):
        nome_arquivo = os.path.basename(img_path)
        print(f"[{i+1}/{total_imgs}] Processando: {nome_arquivo}")
        
        dados_da_imagem = {'Image_Name': nome_arquivo}
        
        for pergunta in PERGUNTAS:
            resposta = perguntar_ao_llava(img_path, pergunta)
            dados_da_imagem[pergunta] = resposta
            
        buffer_dados.append(dados_da_imagem)
        
        if len(buffer_dados) >= INTERVALO_CHECKPOINT:
            salvar_batch_rapido(buffer_dados)
            buffer_dados = [] 

    if buffer_dados:
        salvar_batch_rapido(buffer_dados)

    fim = time.perf_counter()
    tempo_execucao = fim - inicio
    print(f"Tempo total: {tempo_execucao:.4f}s")

if __name__ == "__main__":
    if not os.path.exists(PASTA_IMAGENS):
        print(f"A pasta '{PASTA_IMAGENS}' não existe.")
    else:
        main()