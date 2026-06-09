import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def get_ai_chat_response(df, model_metrics, target_col, user_question, chat_history):
    """Handles conversational chat using Gemini's chat history feature."""
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Context summary to keep Gemini "in the loop"
    context = f"""
    You are an expert Data Scientist. 
    Dataset: {df.shape[0]} rows. Target: {target_col}. 
    ML Performance: {model_metrics}.
    """
    
    # Start the chat session with history
    chat = model.start_chat(history=chat_history)
    
    # Include context in the first prompt
    full_prompt = f"Context: {context}\n\nQuestion: {user_question}"
    
    response = chat.send_message(full_prompt)
    return response.text, chat.history

# Ensure this exact function is at the bottom of GenAI_Insights.py
def draft_full_executive_report(df, metrics, target_col):
    """Generates a structured, long-form report suitable for a PDF."""
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    Write a formal Executive Data Analysis Report based on the following:
    - Target Variable: {target_col}
    - Model Accuracy (R2 Score): {metrics.get('R2 Score')}
    - Data Summary: {df.describe().to_string()}
    
    Structure the report with these sections:
    1. Executive Summary
    2. Data Insights & Trends
    3. Machine Learning Model Performance
    4. Strategic Recommendations
    
    Keep the tone professional and business-focused.
    """
    response = model.generate_content(prompt)
    return response.text