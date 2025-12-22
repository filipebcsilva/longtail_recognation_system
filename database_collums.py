import pandas as pd
import json

def get_data_collums(file):
    try:
        df = pd.read_csv(file, nrows=0)
        collums = df.columns.to_list()
        collums = collums[1:]  
    except:
        collums = []
    return collums

def get_collum_unique_values_string(file):
    dataframe = pd.read_csv(file)
    db_collums = get_data_collums(file) 

    schema_string = {}
    for col in db_collums:
        try:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                min_val = dataframe[col].min()
                max_val = dataframe[col].max()
                
                if hasattr(min_val, 'is_integer') and min_val.is_integer() and max_val.is_integer():
                    min_val = int(min_val)
                    max_val = int(max_val)
                
                schema_string[col] = f"NUMERIC RANGE: {min_val} to {max_val}"
            
            else:
                unique_vals = dataframe[col].unique()
                unique_vals = unique_vals[~pd.isnull(unique_vals)]
            
                try:
                    unique_vals.sort()
                except:
                    pass 
                    
                lista_valores = [val.item() if hasattr(val, 'item') else val for val in unique_vals]
                schema_string[col] = lista_valores
                
        except Exception as e:
            schema_string[col] = f"Error reading values: {str(e)}"

    return json.dumps(schema_string, indent=2, default=str)