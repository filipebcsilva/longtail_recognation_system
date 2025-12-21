import json
from agno.workflow import Step, Workflow, StepOutput,StepInput,Loop
from agents import leitor,saver,analista,gerador_de_perguntas,filtrador
from agno.workflow.condition import Condition
import base64
from agno.media import Image
from LIstIterator import ListIterator
import time
import re
import pandas as pd
from  iterator import ImageIterator
import os

image_iterator = ImageIterator("bdd1")
def needs_vision_model(step_input: StepInput):
    
    data = step_input.get_step_content("Question filter")
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
    print(f"Final data: {final_data}")
    if len(final_data) > 0:
        return True
    else:
        return False

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


def normalize_key(text):
    """Remove espaços e símbolos para comparação (ex: 'Car Count?' vira 'carcount')"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def save_next_image(step_input):
    data = step_input.previous_step_content
    final_data = {}
    file_path = "data.csv"

    if isinstance(data, str):
        clean_text = re.sub(r'```json\s*|\s*```', '', data).strip()
        try:
            final_data = eval(clean_text)
        except:
            try:
                final_data = json.loads(clean_text)
            except:
                try:
                    final_data = eval(clean_text + "}")
                except:
                    print(f"Erro de parse")
                    return
    elif isinstance(data, dict):
        final_data = data

    if not final_data: return
    
    answers = final_data.get("answers")
    new_col_name = final_data.get("question") or final_data.get("query")
    
    if not new_col_name and answers:
        sample = next(iter(answers.values()))
        if isinstance(sample, dict):
             new_col_name = list(sample.keys())[0]
    
    if not new_col_name: new_col_name = "extracted_data"
    final_col_name = new_col_name 
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            existing_headers = pd.read_csv(file_path, index_col=0, nrows=0).columns
            
            norm_new = normalize_key(new_col_name)
            
            for existing in existing_headers:
                if normalize_key(existing) == norm_new:
                    final_col_name = existing 
        except Exception as e:
            print(f"⚠️ Aviso: Não conseguiu ler cabeçalho: {e}")

    source_data = answers if answers else final_data
    processed_source = {}
    
    first_val = next(iter(source_data.values())) if source_data else None
    if isinstance(first_val, dict):
        for k, v in source_data.items():
            processed_source[k] = list(v.values())[0]
    else:
        processed_source = source_data

    new_df = pd.DataFrame.from_dict(processed_source, orient='index', columns=[final_col_name])
    new_df.index.name = 'image_id'

    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            existing_df = pd.read_csv(file_path, index_col=0)
            
            existing_df = existing_df.combine_first(new_df)
            
            if final_col_name in existing_df.columns:
                existing_df.update(new_df)
            
            for col in existing_df.columns:
                try:
                    existing_df[col] = pd.to_numeric(existing_df[col], errors='ignore')
                    if pd.api.types.is_float_dtype(existing_df[col]):
                        existing_df[col] = existing_df[col].round().astype('Int64')
                except: pass

            existing_df.to_csv(file_path, index=True)
            print(f"✅ Salvo em '{final_col_name}'")
        else:
            for col in new_df.columns:
                try:
                     new_df[col] = pd.to_numeric(new_df[col], errors='ignore')
                     if pd.api.types.is_float_dtype(new_df[col]):
                        new_df[col] = new_df[col].round().astype('Int64')
                except: pass
                
            new_df.to_csv(file_path, index=True)
            print(f"Novo arquivo criado com '{final_col_name}'")

    except Exception as e:
        print(f"Não deu para salvar: {e}")


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

gerador_workflow = Workflow(
    name="Gerador pipeline",
    steps= [
        gerador_de_perguntas,
        filtrador,
    ],
    debug_mode = True,
)
