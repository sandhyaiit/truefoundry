import os
import joblib
import pandas as pd
from fastapi import FastAPI

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests

import requests

url = "https://test-object-detect-nikhil-ws.tfy-ctl-euwe1-devtest.devtest.truefoundry.tech/v2/models/test-object-detect/infer"

payload = {
    "id": "string",
    "parameters": {
        "content_type": "string",
        "headers": {}
    },
    "inputs": [
        {
            "name": "inputs",
            "shape": [0],
            "datatype": "string",
            "parameters": {
                "content_type": "string",
                "headers": {}
            },
            "data": "https://placekitten.com/200/300"
        }
    ],
    "outputs": [
        {
            "name": "string",
            "parameters": {
                "content_type": "string",
                "headers": {}
            }
        }
    ]
}
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json()['outputs'][0]['data'])

# Define the request payload schema
class PredictionRequest(BaseModel):
    hf_pipeline: str
    model_deployed_url: str
    inputs: str
    # parameters: dict

# Define the response payload schema
class PredictionResponse(BaseModel):
    prediction: dict

# Create the FastAPI application
app = FastAPI()

xyz = 0

# Define the prediction endpoint
@app.post('/dips/predict')
def predict(request: PredictionRequest):
    hf_pipeline = request.hf_pipeline
    model_deployed_url = request.model_deployed_url
    inputs = request.inputs
    # parameters = request.parameters

    payload = {
        "id": "string",
        "parameters": {
            "content_type": "string",
            "headers": {}
        },
        "inputs": [
            {
                "name": "inputs",
                "shape": [0],
                "datatype": "string",
                "parameters": {
                    "content_type": "string",
                    "headers": {}
                },
                "data": f"{inputs}"
            }
        ],
        "outputs": [
            {
                "name": "string",
                "parameters": {
                    "content_type": "string",
                    "headers": {}
                }
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Make a request to the deployed model endpoint
    # response = requests.post(model_deployed_url, json={'inputs': inputs, 'parameters': parameter})
    # response = requests.post(model_deployed_url, json={'inputs': inputs})
    response = requests.post(model_deployed_url, json=payload, headers=headers)

    # response = requests.post("http://127.0.0.1:8000/", json=a)

    # Extract the prediction from the response
    # prediction = response.json().get('prediction')
    # return {response.json()}
    # print(response.json())
    prediction = response.json()['outputs'][0]['data']
    #
    # print(response.json()['outputs'][0]['data'])
    #
    return {'prediction': prediction}

# Run the application with uvicorn server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port=8000)
