import pandas as pd


def get_data_collums(file):
    try:
        db = pd.read_csv(file)
        collums = db.colums.to_list()
    except:
        collums = []
    return collums