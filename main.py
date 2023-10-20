import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import plotly.express as px

import data_analysis_functions as function

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# page config sets the text and icon that we see on the tab
st.set_page_config(page_icon="✨", page_title="AutoEDA")


# Define custom CSS styles
custom_css = """
<style>
body {
    background-color: #f5f5f5;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    text-align: center;
    padding: 40px;
}

.header {
    font-size: 48px;
    font-weight: bold;
    color: #333;
    margin-bottom: 16px;
}

.tagline {
    font-size: 24px;
    color: #666;
    margin-bottom: 32px;
}

.features {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    margin-bottom: 40px;
}

.feature {
    flex: 1;
    text-align: center;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin: 8px;
    transition: transform 0.3s ease-in-out;
}

.feature:hover {
    transform: scale(1.05);
}

.feature-icon {
    font-size: 36px;
    color: #4CAF50;
}

.feature-title {
    font-size: 18px;
    font-weight: bold;
    margin-top: 16px;
}

.action-button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 16px 32px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.action-button:hover {
    background-color: #45a049;
}

</style>
"""

# Set custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Create the introduction section
st.title("Welcome to AutoEDA")
st.write('<div class="tagline">Unleash the Power of Data with AutoEDA!</div>', unsafe_allow_html=True)

# Highlight the key features
st.write('<div class="features">'
         '<div class="feature">'
         '<div class="feature-icon">📊</div>'
         '<div class="feature-title">Explore datasets interactively.</div>'
         '</div>'
         '<div class="feature">'
         '<div class="feature-icon">🔎</div>'
         '<div class="feature-title">Visualize data with stunning charts.</div>'
         '</div>'
         '<div class="feature">'
         '<div class="feature-icon">🛠️</div>'
         '<div class="feature-title">Preprocess and prepare your data effortlessly.</div>'
         '</div>'
         '</div>', unsafe_allow_html=True)


# Create a Streamlit sidebar
st.sidebar.title("AutoEDA: Automated Exploratory Data Analysis and Processing")

# Create a button in the sidebar to upload CSV
uploaded_file = st.sidebar.file_uploader("Upload Your CSV File Here", type=["csv","xls"])

if uploaded_file:
    df = function.load_data(uploaded_file)


# Display the dataset preview or any other content here
if uploaded_file is None:
    # st.subheader("Welcome to DataExplora!")
    st.markdown("#### Use the sidebar to upload a CSV file and explore your data.")
else:

    navigation=st.sidebar.radio(label="Select Operations",options=['Data Exploration','Data Preprocessing'])
    
    if navigation=='Data Exploration':

    
        if not (navigation):
            st.write("#### Use the sidebar to navigate to either Data Exploration or Data Preprocessing")

        tab1, tab2 = st.tabs(['📊 Dataset Overview :clipboard', "🔎 Data Exploration and Visualization"])
        num_columns, cat_columns = function.categorical_numerical(df)
        
        
        with tab1: # DATASET OVERVIEW TAB
            st.subheader("1. Dataset Preview")
            st.markdown("This section provides an overview of your dataset. You can select the number of rows to display and view the dataset's structure.")
            function.display_dataset_overview(df,cat_columns,num_columns)


            st.subheader("3. Missing Values")
            function.display_missing_values(df)
            
            st.subheader("4. Data Statistics and Visualization")
            function.display_statistics_visualization(df,cat_columns,num_columns)

            st.subheader("5. Data Types")
            function.display_data_types(df)

            st.subheader("Search for a specific column or datatype")
            function.search_column(df)

        with tab2: 

            function.display_individual_feature_distribution(df,num_columns)

            st.subheader("Scatter Plot")
            function.display_scatter_plot_of_two_numeric_features(df,num_columns)


            if len(cat_columns)!=0:
                st.subheader("Categorical Variable Analysis")
                function.categorical_variable_analysis(df,cat_columns)
            else:
                st.info("The dataset does not have any categorical columns")


            st.subheader("Feature Exploration of Numerical Variables")
            if len(num_columns)!=0:
                function.feature_exploration_numerical_variables(df,num_columns)

            else:
                st.warning("The dataset does not contain any numerical variables")

            # Create a bar graph to get relationship between categorical variable and numerical variable
            st.subheader("Categorical and Numerical Variable Analysis")
            if len(num_columns)!=0 and len(cat_columns)!=0:
                function.categorical_numerical_variable_analysis(df,cat_columns,num_columns)
                
            else:
                st.warning("The dataset does not have any numerical variables. Hence Cannot Perform Categorical and Numerical Variable Analysis")
            

    # DATA PREPROCESSING  
    if navigation=='Data Preprocessing':
        st.header("🛠️ Data Preprocessing(To be implemented)")