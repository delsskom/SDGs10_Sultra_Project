import pandas as pd

def get_numeric_columns(df):
    return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

def get_category_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()
