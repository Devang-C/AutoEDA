import streamlit as st
import numpy as np
import pandas as pd

def remove_selected_columns(df,columns_remove):
    return df.drop(columns=columns_remove)

# Create a function to remove rows with missing values in specific columns
def remove_rows_with_missing_data(df, columns):
    if columns:
        df = df.dropna(subset=columns)
        return df

# Create a function to fill missing data with mean, median, or mode (for numerical columns)
def fill_missing_data(df, columns, method):
    for column in columns:
        if method == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif method == 'mode':
            mode_val = df[column].mode().iloc[0]
            df[column].fillna(mode_val, inplace=True)
    return df