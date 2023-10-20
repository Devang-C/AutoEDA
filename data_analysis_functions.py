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

    data_types_df = pd.DataFrame({'Data Type':df.dtypes})
    st.write(data_types_df)

# Function to search for a particular column or particular datatype in the dataset
def search_column(df):
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



## FUNCTIONS FOR TAB2: Data Exploration and Visualization

def display_individual_feature_distribution(df,num_columns):
    st.subheader("Analyze Individual Feature Distribution")
    st.markdown("Here, you can explore individual numerical features, visualize their distributions, and analyze relationships between features.")

    if len(num_columns) == 0:
        st.info("The dataset does not have any numerical columns")
        return

    st.write("#### Understanding Numerical Features")
    feature = st.selectbox(label="Select Numerical Feature", options=num_columns, index=0)
    df_description = df.describe()

    # Display summary statistics
    null_count = df[feature].isnull().sum()
    st.write("Count: ", df_description[feature]['count'])
    st.write("Missing Count: ", null_count)
    st.write("Mean: ", df_description[feature]['mean'])
    st.write("Standard Deviation: ", df_description[feature]['std'])
    st.write("Minimum: ", df_description[feature]['min'])
    st.write("Maximum: ", df_description[feature]['max'])

    # create plots for distribution
    st.subheader("Distribution Plots")
    plot_type = st.selectbox(label="Select Plot Type",options=['Histogram','Scatter Plot','Density Plot','Box Plot'])

    if plot_type=='Histogram':
        fig=px.histogram(df,x=feature,title=f'Histogram of {feature}')

    elif plot_type=='Scatter Plot':
        fig = px.scatter(df,x=feature,y=feature,title=f'Scatter plot of {feature}')

    elif plot_type=='Density Plot':
        fig = px.density_contour(df,x=feature,title=f'Density plot of {feature}')

    elif plot_type=='Box Plot':
        fig = px.box(df,y=feature,title=f'Box plot of {feature}')

    st.plotly_chart(fig,use_container_width=True)


def display_scatter_plot_of_two_numeric_features(df,num_columns):

    if len(num_columns) == 0:
        st.info("The dataset does not have any numerical columns")
        return
    
    if len(num_columns)!=0:
        x_feature = st.selectbox(label="Select X-Axis Feature", options=num_columns, index=0)
        y_feature = st.selectbox(label="Select Y-Axis Feature", options=num_columns, index=1)

        scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f'Scatter Plot: {x_feature} vs {y_feature}')
        st.plotly_chart(scatter_fig, use_container_width=True)



def categorical_variable_analysis(df,cat_columns):

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


def feature_exploration_numerical_variables(df,num_columns):
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


def categorical_numerical_variable_analysis(df,cat_columns,num_columns):
    categorical_feature_1 = st.selectbox(label="Categorical Feature", options=cat_columns)        
    numerical_feature_1 = st.selectbox(label="Numerical Feature", options=num_columns)

# Group by the selected categorical column and calculate the mean of the numerical column
    group_data = df.groupby(categorical_feature_1)[numerical_feature_1].mean().reset_index()

    st.subheader("Relationship between Categorical and Numerical Variables")
    st.write(f"Mean {numerical_feature_1} by {categorical_feature_1}")
    
    # Create a bar chart
    fig = px.bar(group_data, x=categorical_feature_1, y=numerical_feature_1, title=f"{numerical_feature_1} by {categorical_feature_1}")
    st.plotly_chart(fig, use_container_width=True)
