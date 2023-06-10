import io
import json
import sys
import asyncio
from PIL import Image
from fastapi.testclient import TestClient
from main import app, model
import torch

if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

client = TestClient(app)

def test_object_detection():
    with open('test_image.jpg', 'rb') as image_file:
        files = {'file': image_file}
        response = client.post("/objectdetection/", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data

if __name__ == "__main__":
    test_object_detection()