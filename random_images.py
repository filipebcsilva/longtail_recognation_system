import os
import shutil
import random

def copiar_imagens_aleatorias(pasta_origem, pasta_destino, quantidade=100):
    # Lista de extensões que consideramos como imagem
    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # Verifica se a pasta de origem existe
    if not os.path.exists(pasta_origem):
        print(f"Erro: A pasta de origem '{pasta_origem}' não foi encontrada.")
        return

    # Cria a pasta de destino se ela não existir
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
        print(f"Pasta de destino '{pasta_destino}' criada.")

    # Lista todos os arquivos da pasta de origem
    todos_arquivos = os.listdir(pasta_origem)

    # Filtra apenas os arquivos que são imagens (baseado na extensão)
    imagens = [arq for arq in todos_arquivos if arq.lower().endswith(extensoes_validas)]

    total_imagens = len(imagens)
    
    # Verifica se há imagens suficientes
    if total_imagens == 0:
        print("Nenhuma imagem encontrada na pasta de origem.")
        return
    
    # Ajusta a quantidade se houver menos imagens do que o solicitado (evita erro)
    qtd_final = min(quantidade, total_imagens)
    
    if qtd_final < quantidade:
        print(f"Aviso: A pasta contém apenas {total_imagens} imagens. Copiando todas elas.")

    # Seleciona as imagens aleatoriamente
    imagens_selecionadas = random.sample(imagens, qtd_final)

    print(f"Iniciando cópia de {qtd_final} imagens...")

    # Loop para copiar cada imagem
    copias_sucesso = 0
    for imagem in imagens_selecionadas:
        caminho_origem = os.path.join(pasta_origem, imagem)
        caminho_destino = os.path.join(pasta_destino, imagem)

        try:
            # shutil.copy2 preserva os metadados (data de criação, etc)
            shutil.copy2(caminho_origem, caminho_destino)
            copias_sucesso += 1
        except Exception as e:
            print(f"Erro ao copiar {imagem}: {e}")

    print(f"--- Concluído ---")
    print(f"{copias_sucesso} imagens copiadas com sucesso para '{pasta_destino}'.")

# --- CONFIGURAÇÃO ---
# Substitua os caminhos abaixo pelos caminhos reais do seu computador.
# Dica: Use o 'r' antes das aspas para evitar erros com barras no Windows.

destino = "/home/filipe/longtail_recognation_system/bdd50"
origem = "/home/filipe/longtail_recognation_system/bdd100k-dataset/bdd100k/bdd100k/images/100k/train"

# Executa a função
copiar_imagens_aleatorias(origem, destino, quantidade=50)