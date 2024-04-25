# %%
import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

# %%
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
# # Iterate through all the files in indian-foods-80's all folder and list the files which can't be opened as an image ie. are corrupted
# # The location of images are like indian-foods-80/train/aloo-matar/img1 like this
# import os
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# def is_image_file(file_path):
#     try:
#         img = Image.open(file_path)
#         img.verify()
#         return True
#     except (IOError, SyntaxError) as e:
#         return False

# # Iterate through all the folders of indian-foods-80/train
# root_dir = 'indian-foods-80/test'
# corrupted_files = []
# for folder in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder)
#     for file in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file)
#         if not is_image_file(file_path):
#             corrupted_files.append(file_path)
            
# print(f"Number of corrupted files: {len(corrupted_files)}")
    



# %%
# # corrupted_files
# # Delete the corrupted files
# for file in tqdm(corrupted_files):
#     os.remove(file)

# %%
model_checkpoint = "google/vit-base-patch16-224-in21k"

# %%
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="./indian-foods-80", split="train")

# %%
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]

# %%
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

# %%
# Install torchvision
# %pip install torchvision

# %%
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# %%
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

# %%
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# %%
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# %%
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

# %%
print_trainable_parameters(model)

# %%
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# %%
from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    f"{model_name}-finetuned-lora-indian_food",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_names=["labels"],
)

# %%
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# %%
import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# %%
trainer = Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()

# %%
# save model
peft_model_id = "google/vit-base-patch16-224-in21k-lora-indian_food"
trainer.model.save_pretrained(peft_model_id)


# %%
print(trainer.evaluate(val_ds))

# %%



