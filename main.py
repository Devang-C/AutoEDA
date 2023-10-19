import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import plotly.express as px

import data_analysis_functions as function



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

    data_exploration_button = st.sidebar.button("**Data Exploration**")
    data_preprocessing_button = st.sidebar.button("**Data Preprocessing**")
    
    if not (data_exploration_button or data_preprocessing_button):
        st.write("#### Use the sidebar to navigate to either Data Exploration or Data Preprocessing")

    if data_exploration_button:
        
        tab1, tab2 = st.tabs(['📊 Dataset Overview :clipboard', "🔎 Data Exploration and Visualization"])
        num_columns, cat_columns = function.categorical_numerical(df)
        
        
        with tab1: # DATASET OVERVIEW TAB

            function.display_dataset_overview(df,cat_columns,num_columns)
            
            function.display_missing_values(df)
            
            function.display_statistics_visualization(df,cat_columns,num_columns)

            function.display_data_types(df)

            function.search_column(df)

        with tab2: 
            df_description = df.describe()
            st.subheader("Analyze Individual Feature Distribution")
            st.markdown("Here, you can explore individual numerical features, visualize their distributions, and analyze relationships between features.")


            if len(num_columns)!=0:
                st.write("#### Understanding Numerical Features")
                feature = st.selectbox(label="Select Numerical Feature", options=num_columns, index=0)

                # Display summary statistics
                null_count = df[feature].isnull().sum()
                st.write("Count: ", df_description[feature]['count'])
                st.write("Missing Count: ", null_count)
                st.write("Mean: ", df_description[feature]['mean'])
                st.write("Standard Deviation: ", df_description[feature]['std'])
                st.write("Minimum: ", df_description[feature]['min'])
                st.write("Maximum: ", df_description[feature]['max'])

                # Create distribution plots
                st.subheader("Distribution Plots")
                plot_type = st.selectbox(label="Select Plot Type", options=["Histogram", "Scatter Plot","Density Plot", "Box Plot"])

                if plot_type == "Histogram":
                    fig = px.histogram(df, x=feature, title=f'Histogram of {feature}')

                elif plot_type=="Scatter Plot":
                    fig = px.scatter(df,x=feature,y=feature,title=f"Scatter plot of {feature}")
                elif plot_type == "Density Plot":
                    fig = px.density_contour(df, x=feature, title=f'Density Plot of {feature}')
                else:
                    fig = px.box(df, y=feature, title=f'Box Plot of {feature}')

                st.plotly_chart(fig, use_container_width=True)

                # Create scatter plot
                st.subheader("Scatter Plot")
                x_feature = st.selectbox(label="Select X-Axis Feature", options=num_columns, index=0)
                y_feature = st.selectbox(label="Select Y-Axis Feature", options=num_columns, index=1)

                scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f'Scatter Plot: {x_feature} vs {y_feature}')
                st.plotly_chart(scatter_fig, use_container_width=True)

            else:
                st.info("The dataset does not have any numerical columns")


            if len(cat_columns)!=0:

                st.subheader("Categorical Variable Analysis")
                categorical_feature = st.selectbox(label="Select Categorical Feature",options=cat_columns)

                categorical_plot_type = st.selectbox(label="Select Plot Type",options=["Bar Chart","Pie Chart","Stacked Bar Chart","Frequency Count"])

                if categorical_plot_type =="Bar Chart":
                    fig = px.bar(df,x=categorical_feature,title=f"Bar Chart of {categorical_feature}")

                elif categorical_plot_type == "Pie Chart":
                    fig = px.pie(df,names=categorical_feature,title=f"Pie Chart of {categorical_feature}")

                elif categorical_plot_type == "Stacked Bar Chart":
                    st.write("Select a second categorical feature for stacking")
                    second_categorical_feature = st.selectbox(label="Select Second Categorical Feature",options=cat_columns)

                    fig = px.bar(df,x=categorical_feature,color=second_categorical_feature,title=f"Stacked Bar Chart of {categorical_feature} by {second_categorical_feature}")

                elif categorical_plot_type == "Frequency Count":
                    cat_value_counts = df[categorical_feature].value_counts()
                    st.write(f"Frequency Count for {categorical_feature}: ")
                    st.write(cat_value_counts)

                if categorical_plot_type!= "Frequency Count" and fig is not None:
                    st.plotly_chart(fig,use_container_width=True)   


            else:
                st.info("The dataset does not have any categorical columns")


            st.subheader("Feature Exploration of Numerical Variables")
            if len(num_columns)!=0:
                selected_features = st.multiselect("Select Features for Exploration:", num_columns, default=num_columns[:2], key="feature_exploration")

                if len(selected_features) < 2:
                    st.warning("Please select at least two numerical features for exploration.")
                else:
                    st.subheader("Explore Relationships Between Features")

                    # Scatter Plot Matrix
                    if st.button("Generate Scatter Plot Matrix"):
                        scatter_matrix_fig = px.scatter_matrix(df, dimensions=selected_features, title="Scatter Plot Matrix")
                        st.plotly_chart(scatter_matrix_fig, use_container_width=True)

                    # Pair Plot
                    if st.button("Generate Pair Plot"):
                        pair_plot_fig = sns.pairplot(df[selected_features])
                        st.pyplot(pair_plot_fig)

                    # Correlation Heatmap
                    if st.button("Generate Correlation Heatmap"):
                        correlation_matrix = df[selected_features].corr()
                        plt.figure(figsize=(10, 6))
                        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
                        plt.title("Correlation Heatmap")
                        st.pyplot(plt)     

            else:
                st.warning("The dataset does not contain any numerical variables")

            # Create a bar graph to get relationship between categorical variable and numerical variable
            st.subheader("Categorical and Numerical Variable Analysis")
            if len(num_columns)!=0 and len(cat_columns)!=0:
                categorical_feature_1 = st.selectbox(label="Categorical Feature", options=cat_columns)

            
                numerical_feature_1 = st.selectbox(label="Numerical Feature", options=num_columns)

                # Group by the selected categorical column and calculate the mean of the numerical column
                group_data = df.groupby(categorical_feature_1)[numerical_feature_1].mean().reset_index()

                st.subheader("Relationship between Categorical and Numerical Variables")
                st.write(f"Mean {numerical_feature_1} by {categorical_feature_1}")
                
                # Create a bar chart
                fig = px.bar(group_data, x=categorical_feature_1, y=numerical_feature_1, title=f"{numerical_feature_1} by {categorical_feature_1}")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("The dataset does not have any numerical variables. Hence Cannot Perform Categorical and Numerical Variable Analysis")
            

    if data_preprocessing_button:
        st.header("🛠️ Data Preprocessing(To be implemented)")