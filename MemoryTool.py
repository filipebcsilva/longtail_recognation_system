from agno.tools import Toolkit
import pandas as pd
import json
import re
import os
from typing import Union,Dict,Any
       
class MemoryTools(Toolkit):
    def __init__(self, **kwargs):
        super().__init__(name="memory_tools", tools=[self.save_memory,self.read_memory], **kwargs)
    
    def save_memory(self, data: Union[Dict[str, Any], str]):
        """
        Save a dictionary in a csv archive to work as memory
        
        Args:
            data(dict): a dictionary with the format 
            answers:
            {{'Imagem1': {'Pergunta1': 'RespostaG', 'Pergunta2': 'RespostaH'},
            'Imagem5': {'Pergunta1': 'RespostaI', 'Pergunta3': 'RespostaJ'}}
        
        """
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
            
    def read_memory(self):
        """
        Read a csv file and transform into a pandas dataframe
        
        Returns:
            memory (dataframe): a dataframe with images, questions and answers
        """
        file_path = "data.csv"
        memory = pd.read_csv(file_path)
        return memory.to_string()
