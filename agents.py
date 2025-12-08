from agno.agent import Agent
from pydantic import BaseModel,Field
from MemoryTool import MemoryTools
from agno.models.ollama import Ollama
from agno.models.google import Gemini
from agno.tools.python import PythonTools
import os


class LeitorOutuput(BaseModel):
    answers: dict = Field(...,
                               description="A dictionary with the questions and the answer to each question",
                        )
    
text_model = Ollama(
    id="llama3.1:8b",
    options={
            "num_predict": 4096,
            "num_ctx": 8192,
            "temperature":0.0
        }
    )

vision_model = Ollama(
    id="llava:7b",
    options={"temperature": 0.0},
)

code_model = Gemini(
    id = "gemini-2.5-flash",
    provider= "gemini",
    api_key = os.getenv("GEMINI_API_KEY")
)

questions = ["What is the digit on the image"]

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
    description= "You are an AI agent specialized in analyzing words",
    instructions=[
        "TASK: Answer the questions below regarding the input word/image.",
        f"QUESTIONS:\n{questions}",
        
        "OUTPUT FORMAT (CRITICAL):",
        "1. The output must be a dictionary and the keys must be the exact questions listed above.",
        
        "DATA STANDARDS (Apply based on context):",
        "   - QUANTITIES: If the answer is a number, return a PURE INTEGER or FLOAT (e.g., 5, 3.5). Never add units like '5 kg' or '3 people'.",
        "   - BOOLEANS: If the answer is Yes/No, use lowercase 'yes' or 'no'.",
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
      f"CONTEXT: The column names in the dataset are {questions}",
        
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