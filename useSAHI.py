from roboflow import Roboflow
import supervision as sv
import numpy as np
import cv2
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("degen101")
model = project.version(6).model

def callback(image: np.ndarray) -> sv.Detections:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        cv2.imwrite(f.name, image)
        result = model.predict(f.name, confidence=40, overlap=30).json()

    detections = sv.Detections.from_roboflow(result)
    return detections

image = cv2.imread("your_image.jpg")

slicer = sv.InferenceSlicer(callback=callback)

detections = slicer(image=image)

# add your classes from Roboflow, as they appear in the "Classes" section of the "Overview" tab of your model
classes = ["your", "model", "classes"]

prediction_num = len(detections.xyxy)

box_annotator = sv.BoxAnnotator()

annotated_frame = box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    labels=[classes[detections.class_id[i]] for i in range(prediction_num)],
)

sv.plot_image(image=annotated_frame, size=(16, 16))