from fastapi import FastAPI, File, UploadFile
import requests
from typing import Dict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Define the Roboflow API endpoint
ROBOFLOW_API_URL = "https://api.roboflow.com/detect/your-model-id-here"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

@app.post("/count-chips/")
async def count_chips(file: UploadFile = File(...)) -> Dict[str, int]:
    contents = await file.read()
    with open("temp_image.jpg", "wb") as f:
        f.write(contents)

    response = requests.post(
        ROBOFLOW_API_URL,
        files={"file": open("temp_image.jpg", "rb")},
        data={"api_key": ROBOFLOW_API_KEY}
    )

    response_data = response.json()
    chip_count = len(response_data.get("predictions", []))

    return {"chip_count": chip_count}
