from agno.agent import Agent
from pydantic import BaseModel,Field
from typing import List
from agno.models.ollama import Ollama
import os

text_model = Ollama(
    id="llama3.1:8b",
    options={
            "temperature":0.0
        }
    )

class GeradorOutput(BaseModel):
    new_questions: List[str] = Field(
        ..., 
        description="List of questions to ask the vision agent. Return empty list [] if the column already exists in the dataset."
    )
    

questions = ["Number of people","Number of cars","Is it daytime or nighttime","Is there any traffic light","What is the weather conditions","Number of signs"]

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

input = ["Is a child running across the street while it is snowing?"]
    
gerador_de_perguntas.print_response(input=input)