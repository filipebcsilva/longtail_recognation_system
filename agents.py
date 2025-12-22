from agno.agent import Agent
from pydantic import BaseModel,Field,model_validator
from typing import List
from database_collums import get_collum_unique_values_string
from MemoryTool import MemoryTools
from agno.models.ollama import Ollama
from database_collums import get_data_collums
import re
import json
from typing import Any
import ast

def extrair_codigo(texto):

    padrao = r"```(?:python)?\n(.*?)```"
    match = re.search(padrao, texto, re.DOTALL)
    
    if match:
        return match.group(1) 
    else:
        return texto

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


class LeitorOutuput(BaseModel):
    answers: dict = Field(...,
                               description="A dictionary with the questions and the answer to each question",
                        )

class GeradorOutput(BaseModel):
    questions: List[str] = Field(
        ..., 
        description="List of information to be extracted from images."
    )

class FilterOutput(BaseModel):
    output: List[str] = Field(
        ...,
        default_factory=list,
        description="List of questions to ask the vision agent.Return an empty list if all are already covered."
    )
    @model_validator(mode='before')
    @classmethod
    def limpar_sujeira(cls, data: Any) -> Any:
        lista_limpa = extrair_lista_blindada(data)
        
        return {"output": lista_limpa}


text_model = Ollama(
    id="llama3.1:70b-instruct-q6_K",
    options={
            "temperature":0.0
        },
    format="json"
    )

vision_model = Ollama(
    id="llava:34b-v1.6-q8_0",
    options={"temperature": 0.0},
    format='json'
)

code_model = Ollama(
    id = "qwen2.5-coder:32b-instruct-q8_0",
)


questions = get_data_collums("data.csv")

saver = Agent(
    tools=[MemoryTools()],
    model = text_model,
    description = "You are a Agent that uses a tool to save a dictionary on a csv file",
    instructions = [
        "TASK: Call the `save_memory` tool with the received dictionary.",
        
        "CRITICAL RULES FOR DATA INTEGRITY:",
        "1. DO NOT REORDER KEYS.",
        "2. DO NOT ALPHABETIZE.",
        "3. Preserve the exact order of the input dictionary.",
        "4. Treat the input as an immutable binary block.",
        "5. Just pass it exactly as you received it."
    ],
    debug_mode = True,
)

leitor = Agent(
    model = vision_model,
    name = "Word reader",
    description= "You are an AI agent specialized in analyzing images",
    instructions=[
        "TASK: Answer the question on the input regarding the input image.",
        
        "OUTPUT FORMAT (CRITICAL):",
        "1. The output must be a dictionary and the keys must be the exact question",
        "2. The value must be the answer to the question.",
        "3  Do not change the name of the question.",
        "4  Only respond the exact question that was asked.",
        
        "DATA STANDARDS (Apply based on context):",
        "   - QUANTITIES: If the answer is a number, return a PURE INTEGER or FLOAT (e.g., 5, 3.5). Never add units like '5 kg' or '3 people'.",
        "   - BOOLEANS: Return the STRING \"yes\" or \"no\" (lowercase and quoted).",
        "   - TEXT: Keep it concise and lowercase (e.g., 'blue', 'metal').",
        
    ],
    output_schema= LeitorOutuput,
    use_json_mode= True,
    debug_mode= True,
)

analista = Agent(
    model=code_model,
    name="Data Analyst",
    description="You are a Pandas Expert specialized in writing robust, defensive code for data analysis.",
    
   instructions=[
    "### SYSTEM ROLE ###",
    "GOAL: Answer user questions by writing and running Python scripts to EXTRACT image filenames.",
    "IDENTITY: You are an Expert Python Data Analyst. You provide precise lists of matching data.",

    "### INPUT DATA ###",
    "You will receive: USER_QUERY, the name of the COLUMNS and the VALUES that appear at each collum, and the IMAGE DIRECTORY.",
    "If a collum is purely numeric, you will receive its min and max values.",
    "If a collum is categorical, you will receive a list of all unique values it contains.",

    "STRICT TECHNICAL INSTRUCTIONS FOR THE CODE:",
    "1. The index column of the dataframe ALREADY CONTAINS the file extension (e.g., 'photo.jpg').",
    "2. Don't change the collums names. Use them exactly as provided.",
    "3. ALWAYS handle dirty data with robust cleaning techniques.",
    "4. Your final output must be a Python LIST of strings printed to stdout.",

    "### STEP 1: ROBUST DATA LOADING & CLEANING (CRITICAL) ###",
    "0. IMPORT: Import pandas as pd (Do NOT import matplotlib).",
    "1. LOAD: `df = pd.read_csv('data.csv', index_col=0)`.",
    
    "2. ANALYZE THE QUESTION TO DECIDE THE DATA TYPE STRATEGY:",
    
    "   Type A: NUMERIC QUESTIONS (How many, Average, Sum, Max, > 5)",
    "       - ACTION: You MUST convert the column to numeric first.",
    "       - CODE: `df['col'] = pd.to_numeric(df['col'], errors='coerce')`",
    "       - REASON: This handles dirty data/strings by turning them into NaN.",
    
    "   Type B: CATEGORICAL QUESTIONS (Color, Type, Main object, Yes/No)",
    "       - ACTION: Treat as string for comparison.",
    "       - CODE: `df['col'] = df['col'].astype(str).str.lower().str.strip()`",
    "       - LOGIC: Compare against lowercase values (e.g., `df['col'] == 'red'`).",

    "   Type C: COMPLEX FILTERING",
    "       - ACTION: Combine both numeric and categorical cleaning as needed.",
    "       - Combine logic: `df[(df['people'] > 2) & (df['color'] == 'blue')]`.",
    
    "### STEP 2: FILTERING & EXTRACTION (MANDATORY) ###",
    "1. Apply the filtering logic to create `filtered_df`.",
    "2. EXTRACT the index of the `filtered_df` to a list.",
    "   - CODE: `matching_images = filtered_df.index.tolist()`",
    "3. Do NOT limit the number of results. Return ALL matching images.",
    "4. If no images match, return an empty list `[]`.",

    "### STEP 3: FINAL OUTPUT ###",
    "1. ASSIGN the list `matching_images` to a variable named `final_result`.",
    "   - CODE: `final_result = matching_images`",
    "   - DO NOT print anything else.",
    
    "### IMPORTANT RULES ###",
    "1. Return ONLY the executable Python code.",
    "2. DO NOT include explanations, text comments, or markdown blocks (```python)."
    ],
    
    tools=[], 
    debug_mode=True,
)

gerador_de_perguntas = Agent(
    model = text_model, 
    name="Question generator",
    description="Agent responsible for checking if a question has already been answered by the current dataset.",

    instructions = [
    "ROLE: You are a Visual Data Extraction Agent.",
    "GOAL: Translate User Inputs into a precise list of questions/labels to be extracted from images.",
    
    "CRITICAL RULES FOR EXTRACTION:",
    
    "1. DECOMPOSITION LOGIC:",
    "   - Analyze the user input to identify every visual element involved.",
    "   - Example: 'Is a car running a red light?' requires monitoring:",
    "     a) 'Number of cars' (The Object)",
    "     b) 'State of traffic light' (The Context)",
    "     c) 'Vehicle position relative to stop line' (The Action)",
    
    "2. GENERALIZATION RULE:",
    "   - If the user asks about a BROAD CATEGORY (e.g., 'animals', 'obstacles'), create a BROAD question.",
    "   - DO NOT guess specific types unless the user asked for them.",
    "   - BAD: User: 'Any obstacles?' -> Agent: ['Is there a rock?', 'Is there a box?'] (WRONG)",
    "   - GOOD: User: 'Any obstacles?' -> Agent: ['Are there any obstacles?', 'Number of obstacles'] (CORRECT)",
    
    "3. OBJECTIVITY:",
    "   - Keep questions simple and directed at visual recognition.",
    "   - Questions should be phrased as 'Number of [object]', 'Is there [object]', or 'Condition of [object]'.",
    
    "4. OUTPUT FORMAT:",
    "   - Return ONLY a Python list of strings containing the necessary questions.",
    "   - Example: ['Number of cars', 'Is there a traffic light']",
    
    "5. REDUNDANCY FILTER:",
    "   - Do NOT ask both 'Is there [object]?' and 'Number of [object]?'.",
    "   - PREFER 'Number of [object]' as it implies existence.",
    "   - EXCEPTION: Use 'Is there' for abstract concepts (e.g., 'Is there a traffic jam?').",
    
    
    "--- FEW-SHOT EXAMPLES ---",
    
    "Input: 'Can overtaking at night?'",
    "Output: ['Is there a car overtaking', 'What is the time of the day']",
    
    "Input: 'Is the car speeding in the rain?'",
    "Output: ['Is the car speeding', 'What are the weather conditions']",
    
    "Input: 'Is a truck parked in the disabled zone?'",
    "Output: ['Is there a truck parked', 'Is it a disabled parking zone']",
    
    "Input: 'Are pedestrians crossing where there is no crosswalk?'",
    "Output: ['Number of pedestrians crossing', 'Is there a crosswalk']",
    
    "Input: 'Are the car headlights off while driving at night?'",
    "Output: ['Are the car headlights on', 'What is the time of the day']",
    "-------------------------"
    ],
    output_schema=GeradorOutput,
    debug_mode=True,
)


filtrador = Agent(
    model = text_model, 
    name="Question filter",
    description="You are a Semantic Data Filter & Deduplication Specialist.",

    instructions = [
    "You are a Set Difference Engine. You receive two lists: 'CANDIDATES' and 'EXISTING_KNOWLEDGE'.",
    "Your job: Return a list of items from the CANDIDATES that do NOT exist in the EXISTING_KNOWLEDGE.",

    "### STRICT RULES ###",
    "1. EXACT MATCH: If the item is already in the Knowledge Base, REMOVE it.",
    "2. DIFFERENT OBJECTS = NEW DATA: 'Trucks' are different from 'Cars'. 'Motorcycles' are different from 'Bicycles'. If the object type is different, KEEP IT.",
    "3. SPECIFICITY: If the input asks for a specific attribute NOT in the KB (e.g., 'Red Cars' vs 'Cars'), KEEP IT.",
    "4. BROAD TERMS: If the input is broad (e.g., 'Vehicles') and the KB has specific types (e.g., 'Cars', 'Bikes'), KEEP IT.",
    "5. SYNONYMS & PARAPHRASES: Treat synonymous phrases as identical. If the meaning matches an existing item, REMOVE IT.",
    "6. CONTEXT AWARENESS: Consider the overall context. If the input implies a different aspect than what's in the KB, KEEP IT.",
    "7. If the Existing Knowledge is empty, return all Candidates.",
    "8. If all Candidates exist in the Knowledge Base, return an empty list.",
    

   " If 'EXISTING_KNOWLEDGE' is empty, null, or [], YOU MUST RETURN THE 'CANDIDATES' LIST EXACTLY AS IS.",
   " Do not return an empty list if Candidates are provided.",
   
    "### OUTPUT ###",
    " Return a JSON OBJECT with a single key called 'output'.",
    "The value must be the list of kept items.",

    "Example:",
    "{'output': ['Number of people', 'Is there a group']}"
    ],
    output_schema=FilterOutput,
    debug_mode=True,
)


schema_string = get_collum_unique_values_string("data.csv")
analista_input = f"""
    User Query: {"qualquer consulta"}
    Data Columns: {schema_string}
    Image Directory: {"bdd/.100k/images/"}
"""
print(analista_input)