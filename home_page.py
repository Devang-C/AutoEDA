
import streamlit as st



def show_home_page():


    # Key Features
    st.subheader("Key Features")
    st.write("📊 **Interactive Exploration:** Explore your datasets with interactive visualizations.")
    st.write("📈 **Stunning Charts:** Visualize data with beautiful and informative charts.")
    st.write("🛠️ **Effortless Preprocessing:** Streamline data preprocessing and preparation.")

    # Get Started Section
    st.subheader("Get Started with AutoEDA")
    st.write("AutoEDA is your gateway to data analysis and preprocessing. We've simplified the process to help you make the most of your data.")

    # Target Audience
    st.write('<div class="target-audience">'
                '<div class="audience">'
                '<div class="audience-icon">📊</div>'
                '<div class="audience-title">Data Analysts</div>'
                '</div>'
                '<div class="audience">'
                '<div class="audience-icon">🔎</div>'
                '<div class="audience-title">Data Scientists</div>'
                '</div>'
                '<div class="audience">'
                '<div class="audience-icon">🧐</div>'
                '<div class="audience-title">Business Professionals</div>'
                '</div>'
                '<div class="audience">'
                '<div class="audience-icon">📈</div>'
                '<div class="audience-title">Students and Educators</div>'
                '</div>'
                '</div>', unsafe_allow_html=True)

    # Example Dataset
    st.subheader("Try it Out!")
    st.write("Get started by uploading your own dataset or use the example dataset included in sidebar. Select it and let AutoEDA do the rest!")


    # Final Message
    st.write('<div class="thank-you">Start your journey towards data-driven decision-making with AutoEDA!</div>', unsafe_allow_html=True)

    # Add Custom CSS
    st.write('<style>'
            '.target-audience {display: flex; justify-content: space-between; flex-wrap: wrap;}'
            '.audience {flex: 0 1 calc(50% - 10px); background-color: #f6f6f6; border-radius: 10px; margin: 5px; padding: 10px; text-align: center;}'
            '.audience-icon {font-size: 2em;}'
            '.start-button {display: inline-block; margin-top: 20px; background-color: #1E90FF; color: #FFF; padding: 10px 20px; text-align: center; border-radius: 5px; text-decoration: none;}'
            '.thank-you {font-size: 1.5em; margin-top: 20px; text-align: center; color: #555;}'
            '</style>', unsafe_allow_html=True)


def custom_css():
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

    return custom_css