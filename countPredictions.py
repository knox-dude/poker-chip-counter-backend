from roboflow import Roboflow
import supervision as sv
from dotenv import load_dotenv
import os

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("degen101")
model = project.version(6).model

result = model.predict("your_image.jpg", confidence=40, overlap=30).json()

detections = sv.Detections.from_roboflow(result)

print(len(detections))

# filter by class
detections = detections[detections.class_id == 0]
print(len(detections))