from agno.workflow import Workflow,Loop
from agents import gerador_de_perguntas,filtrador,analista
from workflow import get_next_image_step,leitor_step,name_image_step,save_image_step
from iterator import ImageIterator
import pandas as pd
from database_collums import get_data_collums,get_collum_unique_values_string
from agents import extrair_codigo

def longtail_system(query:str,dir_image:str):
    
    db_collums = get_data_collums("data.csv")
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

    candidatos = response_gerador.content.questions  
    kb_atual = db_collums           

    filtrador_input = f"""
    --- EXISTING_KNOWLEDGE ---
    {kb_atual}

    --- CANDIDATES ---
    {candidatos}
    """

    response_filtrador = filtrador.run(filtrador_input)
    
    response_limpa = response_filtrador.content.output
    
    if len(response_limpa) > 0:
        for pergunta in response_limpa:
            image_workflow.run(
            input = pergunta,
            additional_data = {
                "image_iterator": image_iterator,
                "image_name":"",
                "question_name":pergunta,
                
            }
            )
            image_iterator.reset()
    
    
    collums_string = get_collum_unique_values_string("data.csv")
    
    analista_input = f"""
        User Query: {query}
        Data Columns: {collums_string}
        Image Directory: {dir_image}
    """
    
    response = analista.run(analista_input)
    clean_response = extrair_codigo(response.content)
    contexto = {}
    exec(clean_response,globals(),contexto)
    lista_final = contexto.get("final_result",[])
    return lista_final
