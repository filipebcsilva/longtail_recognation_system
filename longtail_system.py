from agno.workflow import Workflow,Loop
from agents import gerador_de_perguntas,filtrador,analista
from workflow import get_next_image_step,leitor_step,name_image_step,save_image_step
from iterator import ImageIterator
import ast
import json
import re
def extrair_lista_blindada(conteudo):
    
    if isinstance(conteudo, list):
        return conteudo

    dados = conteudo

    if isinstance(conteudo, str):
        texto_limpo = re.sub(r'```json\s*|\s*```', '', conteudo).strip()
        
        try:
            dados = json.loads(texto_limpo)
        except json.JSONDecodeError:
            try:
                dados = ast.literal_eval(texto_limpo)
            except:
                print(f"Não foi possível ler o texto: {texto_limpo}")
                return []

    if isinstance(dados, dict):
        for valor in dados.values():
            if isinstance(valor, list):
                return valor
                
    return [] 

def longatail_system(query:str,dir_image:str):
    
    image_iterator = ImageIterator(dir_image)
    
    image_loop = Loop(
            name = "Image loop",
            steps = [get_next_image_step,leitor_step,name_image_step,save_image_step],
            max_iterations = image_iterator.get_size()
        )
    
    image_workflow = Workflow(
        name="vilma pipeline",
        steps= [
            image_loop,
        ],
    debug_mode = True,
    )
    
    response_gerador = gerador_de_perguntas.run(query)

    response_filtrador = filtrador.run(response_gerador.content.questions)
    
    response_limpa = response_filtrador.content.output
    for pergunta in response_limpa:
        image_workflow.run(
        input = pergunta,
        additional_data = {
        "image_iterator": image_iterator,
        "image_name":"",
        }
        )
        image_iterator = ImageIterator(dir_image)
    
    analista.run(query)
    

longatail_system("images with more than three trucks in a rainy or snowy environment during nighttime","bdd1")