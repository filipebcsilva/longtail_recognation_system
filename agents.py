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
    "GOAL:GOAL: Generate only executable Python code that safely extracts image filenames matching the user's query by filtering data.csv, then assigns the result to final_result. "
    "The code will be executed via exec() in a constrained environment—it must never crash, never produce side effects, and must always define final_result as a Python list of strings (even if empty). "
    "Your sole output is raw code. No explanations. No prints. No exceptions..",
    
    "IDENTITY: Expert Python Data Analyst. Code will be executed via `exec()` and `final_result` extracted from context.",
    
    "### CRITICAL CONSTRAINTS ###",
    "1. data.csv is LARGE - code runs server-side. NEVER attempt to load/display full dataframe.",
    "2. Index = image filenames WITH extensions (e.g., 'photo.jpg'). NEVER reset_index(), drop index, or modify it.",
    "3. Column names MUST match schema EXACTLY. Case-sensitive.",
    "4. Code MUST assign to `final_result` in ALL code paths (including errors/empty results).",
    "5. RETURN ONLY raw Python code. NO markdown, NO comments, NO print statements, NO explanations.",
    
    "### STEP 1: DATA LOADING (NON-NEGOTIABLE) ###",
    "import pandas as pd",
    "df = pd.read_csv('data.csv', index_col=0, low_memory=False)",
    
    "### STEP 2: SAFE FILTERING PIPELINE ###",
    "# For EACH column referenced in user query:",
    "#   Validate column exists → if missing: final_result = []; exit pipeline",
    "#   Clean data based on query type:",
    "#      • NUMERIC (>, <, avg, count): pd.to_numeric(..., errors='coerce')",
    "#      • CATEGORICAL (color, object): .fillna('').astype(str).str.lower().str.strip()",
    "#      • BOOLEAN (yes/no): .fillna(False).astype(str).str.lower().isin(['yes','true','1'])",
    "#   Apply filter logic WITHOUT modifying index",
    "#   If ANY step fails → final_result = []; exit gracefully",
    
    "### STEP 3: RESULT EXTRACTION (MANDATORY) ###",
    "# After filtering:",
    "if 'filtered_df' in locals() and not filtered_df.empty:",
    "    final_result = filtered_df.index.tolist()",
    "else:",
    "    final_result = []",
    
    "### ABSOLUTE RULES ###",
    "ALWAYS assign `final_result` before code ends (even if empty list)",
    "NEVER use print(), input(), plt.*, os.*, sys.*, or file I/O",
    "NEVER raise exceptions - fail silently with empty list",
    "NEVER include comments or markdown - pure executable code only",
    "ALWAYS preserve original index format (filenames with extensions)",
    "For categorical matches: lowercase BOTH cleaned column AND query value",
    "NEVER reset_index(), drop index, or create new index",
    "NEVER limit result count - return ALL matches"
    ],
    
    tools=[], 
    debug_mode=True,
)

gerador_de_perguntas = Agent(
    model = text_model, 
    name="Question generator",
    description="""
    You are an expert system designed to process natural language user queries for an image database. 
    Your goal is to deconstruct complex queries into a series of atomic, reusable questions that can be answered individually.
    This allows for efficient caching of answers and minimizes redundant computations when similar queries are made in the future.
    """,

    instructions = [
    """
    You are an intelligent agent designed to process user queries for searching samples in an image database.

    Your goal is to split the user query into atomic questions to avoid repeated heavy computation. This allows the system to cache answers and compose them to answer new queries.

    ### INSTRUCTIONS
    Try to maximize the information gathered in each atomic question to minimize the chance of a new iteration over the database. The answers to the questions should be easily parseable using code.

    ### PREFERRED QUESTION FORMATS
    1. **Quantitative:** "how many of <ITEM-NAME> are there in the image?"
    2. **Categorical:** "what is the <CATEGORY> that best apply to the image among these possibilities <LIST-OF-VALUES>?"
    3. **Binary:** "is there a <ITEM> in the image?"

    ### EXAMPLES

    **Query:** "images without cars with green houses in the background"
    ANSWER: [how many cars are there in the image?, how many green houses in the background are there in the image?]

    **Query:** "images with clear sky in which the largest car is blue"
    ANSWER: [what is the color of the largest car among the following possibilities: [red, green, blue, black, white, silver, pink, orange, brown, purple, other]?,what is the weather condition in the image among the following possibilities: [sunny-dry-or-clear-sky, cloudy, with-snow, with-rain, with-fog, unknown]?]

    **Query:** "images with vehicles parked in the sidewalk"
    ANSWER:[how many vehicles are parked in the sidewalk?]
    
    IMPORTANT:
    Do not decompose questions if it causes a loss of semantics, such as the relationship between the vehicle and the sidewalk).

    ### OUTPUT FORMAT:
    - Return ONLY a Python list of strings containing the necessary questions.
    - Example: ['Number of cars', 'Is there a traffic light']
    """
    ],
    output_schema=GeradorOutput,
    use_json_mode=True,
    debug_mode=True,
)


filtrador = Agent(
    model = text_model, 
    name="Question filter",
    description="You are a Semantic Data Filter & Deduplication Specialist.",

    instructions = [
    """
    You are an expert query optimizer for an image analysis system. Your task is to filter a list of candidate questions against a list of existing questions already stored in the dataset.
    Return ONLY the candidate questions that should be added to the dataset—i.e., those that are NOT already covered by existing questions. 
    
    Some rules you must follow:
    Generalizations: If an existing question is more general and can fully answer a candidate question, FILTER OUT the candidate.
    Example :"what is the weather condition among [rainy, snowy, clear, cloudy, unknown]?" .Filter out candidate "is it raining?" (answer derivable from general weather question).

    Specific objects: KEEP candidate questions that add meaningful specificity NOT captured by existing questions.
    Example 1: Existing = "how many houses are there?". KEEP candidate "how many green houses are there?" (color attribute adds new information).
    Example 2: Example: Existing = "how many cars are there?". KEEP candidate "how many cars parked in sidewalks are there?" (spatial constraint adds new information).

    Object Granularity: Treat object categories as distinct unless explicitly hierarchical in standard taxonomy.
    Example 1: KEEP both "how many cars?" and "how many trucks?" (different vehicle classes).
    Example 2:FILTER OUT "how many sedans?" only if existing question is "how many cars by type among [sedan, SUV, truck, other]?".

    Synonym handling: Treat synonyms as identical ONLY for core object names (e.g., "automobile" = "car"), but preserve distinctions for attributes.
    Example 1:Existing = "how many cars?". FILTER OUT "how many automobiles?".
    Example 2: Example: Existing = "how many red cars?". KEEP "how many crimson cars?" (color shade ambiguity requires separate query).

    Attribute Independence: Questions differing in independent attributes (color, location, state, size) should generally be kept.
    Example 1:KEEP both "how many cars?" and "how many moving cars?" (motion state is new dimension).

    Quantitative vs. Binary: Prefer keeping quantitative questions ("how many...") over binary ("is there...") when the quantitative answer fully determines the binary one.
    Example1:    Existing = "how many pedestrians?". FILTER OUT "are there pedestrians?".

    Output format:
    Return a JSON array containing ONLY the candidate questions that should be added. Preserve exact wording of questions. Do not include explanations.

    General Example:
    1:
    Existing questions: ["how many cars are there in the image?", "what is the weather condition among [rainy, snowy, clear, cloudy, unknown]?"]
    Candidate questions: ["how many cars are there in the image?", "how many trucks are there in the image?", "is it raining?"]

    Output: ["how many trucks are there in the image?"]

    2:
    Existing questions: ["how many houses are there in the image?"]
    Candidate questions: ["how many green houses are there in the image?", "how many houses in the background are there in the image?"]

    Output: ["how many green houses are there in the image?", "how many houses in the background are there in the image?"]
    
    OUTPUT RULES
    Return a JSON OBJECT with a single key called 'output'.
    The value must be the list of kept questions.

    "Example:",
    "{'output': ['Number of people', 'Is there a group']}"
    """
    ],
    output_schema=FilterOutput,
    debug_mode=True,
)

query = ["parking lot at night with snowy or rainy weather"]
response = gerador_de_perguntas.run(input= query)

print(response.content.questions)

