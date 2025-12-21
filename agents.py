from agno.agent import Agent
from pydantic import BaseModel,Field
from typing import List
from MemoryTool import MemoryTools
from agno.models.ollama import Ollama
from database_collums import get_data_collums
from agno.tools.python import PythonTools
import pandas as pd
import os

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
        "GOAL: Answer user questions by writing and running Python scripts using the PythonTools avaiable.",

        "RULES: "
        "1. Try to install the pandas package before writing the code",
        "2. You MUST write the script, don't forget to do that",
        "3. After installing the pandas package, folow the steps:"
        
        "STRATEGY FOR MIXED DATA TYPES (Critical):",
        
        "1. LOADING:",
        "   - `df = pd.read_csv('data.csv', index_col=0)`",
        
        "2. ANALYZE THE QUESTION TO DECIDE THE DATA TYPE:",
        
        "   Type A: NUMERIC QUESTIONS (How many, Average, Sum, Max, > 5)",
        "       - ACTION: You MUST convert the column to numeric first.",
        "       - CODE: `df['col'] = pd.to_numeric(df['col'], errors='coerce')`",
        "       - REASON: This handles dirty data (e.g., text mixed with numbers) by turning it into NaN.",
        
        "   Type B: CATEGORICAL QUESTIONS (Color, Type, Main object, Yes/No)",
        "       - ACTION: Treat as string. Do NOT use to_numeric.",
        "       - CODE: `df['col'] = df['col'].astype(str).str.lower().str.strip()`",
        "       - OPS: Use `.value_counts()`, `.unique()` or string filtering (`==`).",
        
        "3. COMPLEX FILTERING (Combining types):",
        "   - Example: 'Show images with > 2 people AND Blue color'.",
        "   - Logic: Apply Type A logic to the 'people' column AND Type B logic to the 'color' column.",
        "   - Code: `df[(df['people'] > 2) & (df['color'] == 'blue')]`.",
        
        "4. OUTPUT:",
        "   - Always use `print()` to display the result.",

    ],
    
    tools=[PythonTools()], 
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
    "### SYSTEM ROLE ###",
    "You are a STRICT DATA FILTER ENGINE. You are NOT a chat assistant.",
    "Your Goal: Perform a mathematical Set Difference: (USER_INPUT - KNOWLEDGE_BASE).",

    "### KNOWLEDGE BASE (Columns we already have) ###",
    f"{questions}", 
    "--------------------------------------------------",

    "### THE INPUT DATA ###",
    "The user will send you a list of strings (the CANDIDATE LIST) in the next message.",
    "You must process THAT list against the Knowledge Base above.",
    "If the Knowledge Base is empty, just return the same string as output"

    "### STRICT EXECUTION RULES ###",
    "1. COPY-PASTE ONLY: If a candidate from the user's list is NOT in the Knowledge Base, return it EXACTLY as written.",
    "2. EXACT MATCH FILTER: If a candidate IS in the Knowledge Base, discard it.",
    "3. EMPTY KB RULE: If the Knowledge Base above is empty [], return the user's list exactly as is.",
    "4. NO HALLUCINATIONS: Do not change words. Do not fix typos. Do not add adjectives. If input is 'Trucks', output 'Trucks'.",

    "### OUTPUT FORMAT ###",
    "Return ONLY a Python list of strings. Example: ['Item A', 'Item B'].",
    "Do not write 'Here is the list'. Just the list."
    ],
    output_schema=FilterOutput,
    debug_mode=True,
)
