import json
from agno.workflow import Step, Workflow, StepOutput,StepInput,Loop
from agents import leitor,saver,analista
import base64
from agno.media import Image
from  iterator import ImageIterator
import os

iterator = ImageIterator("mnist_images")

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

vilma_workflow = Workflow(
    name="Pandas pipeline",
    steps= [
        Loop(
            name = "Image loop",
            steps = [get_next_image_step,leitor_step,name_image_step,saver_step],
            max_iterations = iterator.get_size()
        ),
        get_user_questions_step,
        analista_step
    ],
    debug_mode = True,
)

input = [
    "What is the quantity of each digit in the dataset?",
]

vilma_workflow.run(
    input = "Execute o workflow",
    additional_data = {
        "iterator":iterator,
        "image_name":"",
        "user_questions":input
    }
)
