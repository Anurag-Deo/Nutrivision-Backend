import requests
import json
from fastapi import FastAPI, UploadFile, File
import uvicorn
import transformers
import accelerate
import peft
from PIL import ImageFile,Image
from datasets import load_dataset
from transformers import AutoImageProcessor,AutoModelForImageClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

ImageFile.LOAD_TRUNCATED_IMAGES = True

model_checkpoint = "/DATA/anurag_2101ai04/Assignment/CV/google/vit-base-patch16-224-in21k-lora-indian_food/"

dataset = load_dataset("imagefolder", data_dir="./indian-foods-80", split="train")

labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

inference_model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

# Create a fastapi app with a single endpoint /detect which takes an image file and runs some code to get the label and then return it

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.post("/detect")
async def get_image_label(image: UploadFile = File(...)):
    # Save the image to a file
    with open('image.jpg', 'wb') as f:
        f.write(image.file.read())
    # Run the code to get the label
    # Take an image from the indian-foods-80 dataset
    image_path = "image.jpg"
    image = Image.open(image_path)

    encoding = image_processor(image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = inference_model(**encoding)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    # print("Predicted class:", inference_model.config.id2label[predicted_class_idx])
    # labels.append(inference_model.config.id2label[predicted_class_idx])
    response = {'dish':inference_model.config.id2label[predicted_class_idx]}
        
    # Return the label
    return response

# Run the app using uvicorn
uvicorn.run(app, host="localhost", port=8003)