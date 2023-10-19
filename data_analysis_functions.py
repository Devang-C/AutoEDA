''' This file contains all the functions that are used in the main file. 
This is so as to reduce the clutter in the main file and isolate the core functionalites of the application in seprate file
'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px

# Function to load the csv data to a dataframe
def load_data(file):
    return pd.read_csv(file)

# Function to find categorical and numerical columns/variables in dataset
def categorical_numerical(df):
    num_columns,cat_columns = [],[]
    for col in df.columns:
        if len(df[col].unique()) <= 30 or df[col].dtype== np.object_:
            cat_columns.append(col.strip())

        else:
            num_columns.append(col.strip())

    return num_columns,cat_columns


# Function to display dataset overview
def display_dataset_overview(df,cat_columns,num_columns):
    st.subheader("1. Dataset Preview")
    st.markdown("This section provides an overview of your dataset. You can select the number of rows to display and view the dataset's structure.")
    display_rows = st.slider("Display Rows", 1, len(df), len(df) if len(df) < 20 else 20)

    st.write(df.head(display_rows))

    st.subheader("2. Dataset Overview")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(f"**Duplicates:** {df.shape[0] - df.drop_duplicates().shape[0]}")
    st.write(f"**Categorical Columns:** {len(cat_columns)}")
    st.write(cat_columns)
    st.write(f"**Numerical Columns:** {len(num_columns)}")
    st.write(num_columns)
    

# Function to find the missing values in the dataset
def display_missing_values(df):
    st.subheader("3. Missing Values")
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    if not missing_data.empty:
        st.write("Missing Data Summary:")
        st.write(missing_data)

    else:
        st.info("No Missing Value present in the Dataset")

# Function to display basic statistics and visualizations about the dataset
def display_statistics_visualization(df,cat_columns,num_columns):
    st.subheader("4. Data Statistics and Visualization")
    st.write("Summary Statistics for Numerical Columns")

    if len(num_columns)!=0:
        num_df = df[num_columns]
        st.write(num_df.describe())

    else:
        st.info("The dataset does not have any numerical columns")

    
    st.write("Statistics for Categorical Columns")
    if len(cat_columns)!=0:
        num_cat_columns = st.number_input("Select the number of categorical columns to visualize:",min_value=1,max_value=len(cat_columns))
        selected_cat_columns = st.multiselect("Select the Categorical Columns for bar chart",cat_columns,cat_columns[:num_cat_columns])

        for column in selected_cat_columns:
            st.write(f"**{column}**")
            value_counts = df[column].value_counts()
            st.bar_chart(value_counts)

            # display the value count in tabular format
            st.write(f"Value Count for {column}")
            value_counts_table = df[column].value_counts().reset_index()
            value_counts_table.columns = ['Value','Count']
            st.write(value_counts_table)

    else:
        st.info("The dataset does not have any categorical columns")

# Funciton to display the datatypes
def display_data_types(df):
    st.subheader("5. Data Types")

    data_types_df = pd.DataFrame({'Data Type':df.dtypes})
    st.write(data_types_df)

# Function to search for a particular column or particular datatype in the dataset
def search_column(df):
    st.subheader("Search for a specific column or datatype")
    search_query = st.text_input("Search for a column:")

    selected_data_type = st.selectbox("Filter by Data Type:", ['All'] + df.dtypes.unique().tolist())

    # Apply filters to the DataFrame
    filtered_df = df.copy()

    # Filter by search query
    if search_query:
        filtered_df = filtered_df.loc[:, filtered_df.columns.str.contains(search_query, case=False)]

    # Filter by data type
    if selected_data_type != 'All':
        filtered_df = filtered_df.select_dtypes(include=[selected_data_type])

    # Display the filtered DataFrame
    st.write(filtered_df)