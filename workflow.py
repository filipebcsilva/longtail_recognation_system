import json
from agno.workflow import Step, Workflow, StepOutput,StepInput,Loop
from agents import leitor,saver,analista,gerador_de_perguntas,filtrador
from agno.workflow.condition import Condition
import base64
from agno.media import Image
import time
import re
import pandas as pd
from  iterator import ImageIterator
import os

iterator = ImageIterator("bdd100")

def needs_vision_model(step_input: StepInput):
    
    data = step_input.previous_step_content
    
    final_data = []
    
    if isinstance(data, str):
        clean_text = re.sub(r'```json\s*|\s*```', '', data).strip()
        
        try:
            final_data = eval(clean_text)
        except (ValueError, SyntaxError):
            try:
                final_data = json.loads(clean_text)
            except json.JSONDecodeError:
                try:
                    final_data = eval(clean_text + "}")
                except:
                    print(f"Não foi possível ler os dados {clean_text}")
    elif isinstance(data, list):
        final_data = data
    
    if len(final_data) > 0:
        return True
    else:
        return False

def fetch_next_image(step_input: StepInput) -> StepOutput:
    iterator = step_input.additional_data.get("iterator")
    
    image_path = iterator.get_next()
    
    if image_path:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        
            image = Image(
            content=base64_string 
            )
        
        step_input.additional_data["image_name"] = os.path.basename(image_path)

        return StepOutput(
            step_name="fetch_image",
            images = [image],
            success=True,
        )

def save_next_image(step_input: StepInput):
    data = step_input.previous_step_content
    
    final_data = {}

    if isinstance(data, str):
        clean_text = re.sub(r'```json\s*|\s*```', '', data).strip()
        
        try:
            final_data = eval(clean_text)
        except (ValueError, SyntaxError):
            try:
                final_data = json.loads(clean_text)
            except json.JSONDecodeError:
                try:
                    final_data = eval(clean_text + "}")
                except:
                    print(f"Não foi possível ler os dados {clean_text}")


    elif isinstance(data, dict):
        final_data = data

    answers = final_data.get("answers")
    
    source_data = answers if answers else final_data
    
    temp = pd.DataFrame.from_dict(source_data, orient='index')
    
    file_path = "data.csv"
    
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    try:
        temp.to_csv(file_path, mode='a', index=True, header=not file_exists)
    except Exception as e:
        print(f"Erro ao salvar arquivo")
            
    
def insert_input_questions(step_input: StepInput) -> StepOutput:

    return StepOutput(
        step_name="insert_user_question",
        content= step_input.additional_data["user_questions"],
        success=True,
    )


def insert_name_image(step_input: StepInput) -> StepOutput:
    message = step_input.previous_step_content

    name_image = step_input.additional_data.get("image_name")
    answers = {
        "answers": {
            name_image: message.answers
        }
    }
    
    answers_str = json.dumps(answers,ensure_ascii=False)
    return StepOutput(
        step_name="insert_name_images", content=answers_str, success=True,
    )

leitor_step = Step(
    name = "Leitor",
    agent = leitor
)

saver_step = Step(
    name = "Registrador",
    agent = saver
)

analista_step = Step(
    name = "Code generator",
    agent = analista
)

name_image_step = Step(
    name = "Insere nome da imagem",
    executor = insert_name_image
)

get_next_image_step = Step(
    name = "Pega a proxima imagem",
    executor = fetch_next_image
)

get_user_questions_step = Step(
    name = "Insere as perguntas do usuario",
    executor = insert_input_questions
)

save_image_step = Step(
    name = "Image Saver",
    executor = save_next_image
)

image_loop = Loop(
            name = "Image loop",
            steps = [get_next_image_step,leitor_step,name_image_step,save_image_step],
            max_iterations = iterator.get_size()
        )



vilma_workflow = Workflow(
    name="Pandas pipeline",
    steps= [
        image_loop,
        get_user_questions_step,
        analista_step
    ],
    debug_mode = True,
)

input = [
    "What is the quantity of each digit in the dataset?",
]

inicio = time.perf_counter()


vilma_workflow.run(
    input = input,
    additional_data = {
        "iterator":iterator,
        "image_name":"",
        "user_questions":input
    }
)

fim = time.perf_counter()

tempo_execucao = fim - inicio

print(f"O código levou {tempo_execucao:.4f} segundos para rodar.")