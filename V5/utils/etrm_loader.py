import pandas as pd

def load_etrm_data(path: str):
    df = pd.read_excel(path)
    return df
