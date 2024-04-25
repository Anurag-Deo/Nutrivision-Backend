#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
GPT_API_KEY = ""
GEMINI_API_KEY = ""
ANTHROPIC_API_KEY = ""

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# food = pd.read_csv('indian_food.csv')
# Read the file List of Indian Foods.txt and store the content in a variable food
with open('List of Indian Foods.txt', 'r') as file:
    food = file.read()


model_gpt=ChatOpenAI(model="gpt-3.5-turbo", api_key=GPT_API_KEY)
model_gemini = ChatGoogleGenerativeAI(model="gemini-1.0-ultra-latest" , google_api_key=GEMINI_API_KEY)
model_claude = ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key=ANTHROPIC_API_KEY)
# model = ChatGoogleGenerativeAI(model="gemini-pro" , google_api_key="AIzaSyDSs57SWYjmlZTegkFgg2QLBAsHaAwcmXM")
add_routes(app, model_gemini, path="/chat_gemini")
add_routes(app, model_claude, path="/chat_claude")
add_routes(app, model_gpt, path="/chat_gpt")

prompt = ChatPromptTemplate.from_template("""
    You have the entire knowledge about the nutritional values regarding any of the indian dishes. I will give you the name of the dish and you have to give me the nutritional values in the form of output shown below. For each of the nutrient in the output you have to tell the amount and the daily value(%). Also try to give the approximate quantity that i should consume so that it will not hurt my health much. MAKE SURE THE VALUES YOU GIVE ARE CORRECT UPTO YOUR KNOWLEDGE.
    The input should be like:
    Dish: DISH NAME
    The output should be like:
    Nutritional Value: 
    Calories: AMOUNT (DAILY VALUE)
    Protein: AMOUNT (DAILY VALUE)
    Carbohydrates: AMOUNT (DAILY VALUE)
    Fats: AMOUNT (DAILY VALUE)
    Fiber: AMOUNT (DAILY VALUE)
    Sugars: AMOUNT (DAILY VALUE)
    Cholesterol: AMOUNT (DAILY VALUE)
    Sodium: AMOUNT (DAILY VALUE)
    Iron: AMOUNT (DAILY VALUE)
    Calcium: AMOUNT (DAILY VALUE)
    
    Approximate Serving: SERVING VALUE IN CUPS (VALUE IN GRAMS OR ML)

    AVOID ANY OTHER TEXT IN THE OUTPUT AND MAKE SURE THAT THE OUTPUT IS IN THE FORMAT SHOWN ABOVE. THERE MUST BE PARENTHESES IN THE DAILY VALUE FOR EACH OF THE NUTRIENTS.

    Dish : {dish}
""")

# prompt = ChatPromptTemplate.from_template(prompt_text)
add_routes(
    app,
    prompt | model_gemini,
    path="/info_gemini",
)

add_routes(
    app,
    prompt | model_claude,
    path="/info_claude",
)

add_routes(
    app,
    prompt | model_gpt,
    path="/info_gpt",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8002)