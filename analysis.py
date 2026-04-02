import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-2.5-flash')

def generate_insights(results_df):
    prompt = f"""
    You are a data scientist analyzing the results of different machine learning models. Here are the results:
    {results_df.to_string()}
    
    1. Identity the best performing model based on the metrics provided.
    2. Explain why that model is the best.
    3. Summarize the performance of all models.    
    """
    
    response = model.generate_content(prompt)
    return response.text

def suggest_improvements(results_df):
    prompt = f"""
    You are a data scientist analyzing the results of different machine learning models. Here are the results:
    {results_df.to_string()}
    
    Suggest:
    - Ways to improve the performance of the models.
    - Better algorithms if needed.
    - Data preprocessing improvements.
    """
    
    response = model.generate_content(prompt)
    return response.text