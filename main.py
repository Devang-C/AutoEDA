import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import plotly.express as px
from streamlit_option_menu import option_menu
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function




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

# Create a Streamlit sidebar
st.sidebar.title("AutoEDA: Automated Exploratory Data Analysis and Processing")

selected = option_menu(
    menu_title=None,
    options=['Home','Data Exploration','Data Preprocessing'],
    icons=['house-heart','bar-chart-fill','hammer'],
    orientation='horizontal'
)

if selected=='Home':

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


# Create a button in the sidebar to upload CSV
uploaded_file = st.sidebar.file_uploader("Upload Your CSV File Here", type=["csv","xls"])

if uploaded_file:
    df = function.load_data(uploaded_file)

    # get a copy of original df from the session state or create a new one. this is for preprocessing purposes
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df.copy()

    # new_df = st.session_state.new_df


# Display the dataset preview or any other content here
if uploaded_file is None:
    # st.subheader("Welcome to DataExplora!")
    st.markdown("#### Use the sidebar to upload a CSV file and explore your data.")
else:
    
    if selected=='Data Exploration':

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
    if selected=='Data Preprocessing':
        # st.header("🛠️ Data Preprocessing")


        # REMOVING UNWANTED COLUMNS
        st.subheader("Remove Unwanted Columns")
        columns_to_remove = st.multiselect(label='Select Columns to Remove',options=st.session_state.new_df.columns)

        if st.button("Remove Selected Columns"):
            if columns_to_remove:
                st.session_state.new_df = preprocessing_function.remove_selected_columns(st.session_state.new_df,columns_to_remove)
                st.success("Selected Columns Removed Sucessfully")
                
        st.dataframe(st.session_state.new_df)
       

       # Handle missing values in the dataset
        st.subheader("Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()

        if missing_count.any():

            selected_missing_option = st.selectbox(
                "Select how to handle missing data:",
                ["Remove Rows in Selected Columns", "Fill Missing Data in Selected Columns (Numerical Only)"]
            )

            if selected_missing_option == "Remove Rows in Selected Columns":
                columns_to_remove_missing = st.multiselect("Select columns to remove rows with missing data", options=st.session_state.new_df.columns)
                if st.button("Remove Rows with Missing Data"):
                    st.session_state.new_df = preprocessing_function.remove_rows_with_missing_data(st.session_state.new_df, columns_to_remove_missing)
                    st.success("Rows with missing data removed successfully.")

            elif selected_missing_option == "Fill Missing Data in Selected Columns (Numerical Only)":
                numerical_columns_to_fill = st.multiselect("Select numerical columns to fill missing data", options=st.session_state.new_df.select_dtypes(include=['number']).columns)
                fill_method = st.selectbox("Select fill method:", ["mean", "median", "mode"])
                if st.button("Fill Missing Data"):
                    if numerical_columns_to_fill:
                        st.session_state.new_df = preprocessing_function.fill_missing_data(st.session_state.new_df, numerical_columns_to_fill, fill_method)
                        st.success(f"Missing data in numerical columns filled with {fill_method} successfully.")

                    else:
                        st.warning("Please select a column to fill in the missing data")

            function.display_missing_values(st.session_state.new_df)

        else:
            st.info("The dataset does not contain any missing values")
            

    