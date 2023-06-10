import io
import json
from PIL import Image
from fastapi import File, FastAPI
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = "https://ic.wampi.ru/2023/06/05/photo_2023-06-05_21-12-59.jpg"
results = model(img)
print(results.pandas().xyxy[0].to_json(orient="records"))

app = FastAPI()

@app.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    return {"result": results_json}
