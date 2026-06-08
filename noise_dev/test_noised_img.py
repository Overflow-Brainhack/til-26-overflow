from ultralytics import YOLO, RTDETR
import random
import json
import base64
import requests
import io
from PIL import Image

# model = YOLO("noise_dev/yolo11x-finetuned.pt")
model = RTDETR("noise/models/rtdetr-l-70.pt")

img = random.randint(1, 1000)
print(f"Testing image {img}")

with open(
    f"/home/shadowmachete/dev/til-26-overflow/data/cv/images/{img}.jpg",
    "rb",
) as img_file:
    img_data = img_file.read()
img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")

result = model(img_pil, imgsz=1280, rect=True, conf=0.5)
result[0].show()

response = requests.post(
    "http://localhost:5003/noise",
    data=json.dumps(
        {
            "instances": [
                {
                    "key": 0,
                    "b64": base64.b64encode(img_data).decode("ascii"),
                }
            ],
        }
    ),
)
noised_img = response.json()["predictions"][0]

img_bytes = base64.b64decode(noised_img)

img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
img.show()

result = model(img, imgsz=1280, rect=True, conf=0.5)
result[0].show()
