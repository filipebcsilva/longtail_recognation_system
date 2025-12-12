import ollama
import os
import glob
import pandas as pd
import time

PASTA_IMAGENS = "./bdd100"
ARQUIVO_SAIDA = "new_resultado_bdd100.csv"
MODELO = "llava:7b"
INTERVALO_CHECKPOINT = 50 

PERGUNTAS = [
    "Number of people", "Number of cars", "Number of buses or trucks", 
    "Is there any red traffic light", "Is there any green traffic light", 
    "Is there any traffic signs","Number of rider in motorcyles or bikes",
    "What is the weather (clear/overcast/rainy/snowy/foggy/partly cloud/undefined)",
    "Time of the day (daytime/night/dawn)",
    "Type of scene (city stree/residential/highway/parking lot/tunnel/gas station)"
]

def perguntar_ao_llava(caminho_imagem, pergunta):
    prompt = f"""
    Look at the image and answer the question.
    Question: {pergunta}
    
    RULES:
    - Answer ONLY with the result. No sentences.
    - If it's a count question or a quantity question, return ONLY the integer (e.g., 3).
    - If it's a count or quantity question and the answer is zero, return 0. Don't write texts like 'No people'
    - If it's yes/no, return ONLY "yes" or "no".
    - If it's text,keep it concise and lowercase (e.g., 'blue', 'metal').",
    """
    
    try:
        response = ollama.generate(
            model=MODELO,
            prompt=prompt,
            images=[caminho_imagem],
            options={
                "temperature": 0.0, 
                "num_ctx": 4096,
            },
        )
        return response['response'].strip()
    except Exception as e:
        print(f"Erro na api: {e}")
        return "error"

def salvar_batch_rapido(lista_novos_dados):
    if not lista_novos_dados:
        return

    print(f"Salvando Checkpoint ({len(lista_novos_dados)} imagens)")

    df_novo_batch = pd.DataFrame(lista_novos_dados)

    cols_ordenadas = ['Image_Name'] + PERGUNTAS
    
    df_novo_batch = df_novo_batch.reindex(columns=cols_ordenadas)
    
    arquivo_existe = os.path.exists(ARQUIVO_SAIDA) and os.path.getsize(ARQUIVO_SAIDA) > 0
    escrever_cabecalho = not arquivo_existe
    
    df_novo_batch.to_csv(ARQUIVO_SAIDA, mode='a', header=escrever_cabecalho, index=False)
    with open("last_image.txt", 'w') as f:
        f.write(lista_novos_dados[-1]['Image_Name']  + "\n")
        index = len(lista_novos_dados) - 1
        f.write(str(index))

def main():
    imagens = []
    imagens.extend(glob.glob(os.path.join(PASTA_IMAGENS, "*.jpg")))
    
    imagens = sorted(list(set(imagens)))
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