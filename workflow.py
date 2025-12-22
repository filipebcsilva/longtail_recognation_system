import json
from agno.workflow import Step, Workflow, StepOutput,StepInput,Loop
from agents import leitor,saver,analista,gerador_de_perguntas,filtrador

import base64
from agno.media import Image
import re
import pandas as pd
from  iterator import ImageIterator
import os

image_iterator = ImageIterator("bdd1")

def fetch_next_image(step_input: StepInput) -> StepOutput:
    iterator = step_input.additional_data.get("image_iterator")
    
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

def get_next_question(step_input: StepInput) -> StepOutput:
    iterator = step_input.additional_data.get("question_iterator")
    
    question = iterator.get_next()
    
    return StepOutput(
        step_name="get_next_image",
        content= question,
        success=True,
    )

def save_next_image(step_input):
    forced_col_name =  step_input.additional_data.get("question_name")
    
    if not forced_col_name and hasattr(step_input, 'inputs') and isinstance(step_input.inputs, dict):
        forced_col_name = step_input.inputs.get('original_question')

    data = step_input.previous_step_content
    file_path = "data.csv"
    final_data = {}

    if isinstance(data, str):
        clean_text = re.sub(r'```json\s*|\s*```', '', data).strip()
        try:
            final_data = json.loads(clean_text)
        except:
            try:
                final_data = eval(clean_text)
            except:
                print("Erro crítico: Não consegui ler o JSON de resposta.")
                return
    elif isinstance(data, dict):
        final_data = data

    if not final_data: return

    if forced_col_name:
        final_col_name = forced_col_name
        print(f"Usando nome forçado da pergunta: '{final_col_name}'")
    else:
        print("Aviso: Nome original não encontrado no input. Tentando extrair do JSON...")
        final_col_name = final_data.get("question") or final_data.get("query") or "extracted_data"

    answers = final_data.get("answers", final_data)
    processed_source = {}

    first_val = next(iter(answers.values())) if answers else None
    
    if isinstance(first_val, dict):
        for k, v in answers.items():
            val = v.get(final_col_name)
            if val is None:
                 val = list(v.values())[0]
            processed_source[k] = val
    else:
        processed_source = answers

    new_df = pd.DataFrame.from_dict(processed_source, orient='index', columns=[final_col_name])
    new_df.index.name = 'Image_Name'

    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            existing_df = pd.read_csv(file_path, index_col=0)
            existing_df = existing_df.combine_first(new_df)
            
            if final_col_name in existing_df.columns:
                existing_df.update(new_df)
        
            try:
                existing_df[final_col_name] = pd.to_numeric(existing_df[final_col_name], errors='ignore')
            except: pass
                
            existing_df.to_csv(file_path, index=True)
            print(f"Sucesso! Dados salvos na coluna '{final_col_name}'")
        else:
            new_df.to_csv(file_path, index=True)
            print(f"Novo arquivo criado com a coluna '{final_col_name}'")

    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")

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
get_next_question_step = Step(
    name = "next question step",
    executor = get_next_question
)
save_image_step = Step(
    name = "Image Saver",
    executor = save_next_image
)
