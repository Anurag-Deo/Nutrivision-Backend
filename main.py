import requests
import json
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

MODEL = "gemini" # Can be gemini or gpt or claude

def get_nutritional_data(facts):
    x=facts.split('Approximate Serving:')
    serving = x[1].split('\n')[0].strip()
    nutritional_values = x[0].split('Nutritional Value:')[1][1:]
    nutrients = nutritional_values.split('\n')
    nutritional_data = {}
    for nutrient in nutrients:
        if nutrient:
            nutritional_data[nutrient.split(':')[0].strip()] = {'value':nutrient.split(':')[1].strip().split(' ')[0].strip(),'percentage':nutrient.split(':')[1].strip().split(' ')[1][1:-1].strip()}
    return nutritional_data,serving

# Create a fastapi app with a single endpoint /nutritional_info post method that takes a json input with a key dish and returns the nutritional data and serving size of the dish

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.post("/nutritional_info")
async def get_nutritional_info(data: dict):
    dish = data['dish']
    if MODEL=='gemini':
        response=requests.post(
            "http://localhost:8002/info_gemini/invoke",
            json={'input':{'dish':dish}})
    elif MODEL=='gpt':
        response=requests.post(
            "http://localhost:8002/info_gpt/invoke",
            json={'input':{'dish':dish}})
    elif MODEL=='claude':
        response=requests.post(
            "http://localhost:8002/info_claude/invoke",
            json={'input':{'dish':dish}})
    # response=requests.post(
    #     "http://localhost:8002/info_gemini/invoke",
    #     json={'input':{'dish':dish}})
    facts = response.json()['output']['content']
    print(facts)
    nutritional_data,serving = get_nutritional_data(facts)
    # Convert the nutritional_data to json for easy parsing with javascript
    # nutritional_data = json.dumps(nutritional_data)
    return {'nutritional_data':nutritional_data,'serving':serving}

# Run the app using uvicorn
uvicorn.run(app, host="localhost", port=8001)