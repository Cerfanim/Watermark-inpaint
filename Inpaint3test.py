import os
import gc
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama
from PIL import Image
import torchvision.transforms as transforms

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv8 model
yolo_model = YOLO("F:\\LAMAINPAINT\\watermark.pt").to(device)

# Set YOLOv8 model to evaluation mode
yolo_model.model.eval()

# Enable FP16 for YOLOv8
yolo_model.fuse()
yolo_model.model.half()

# Load LAMA model
simple_lama = SimpleLama(device=device)
simple_lama.model.eval()

# Input image folder path
img_folder = "F:/STD/nasral"

# Output folder path
output_folder = "F:/LAMAINPAINT/output"
os.makedirs(output_folder, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Process images one by one
img_paths = [
    os.path.join(img_folder, filename)
    for filename in os.listdir(img_folder)
    if filename.lower().endswith((".jpg", ".png"))
]

for img_path in img_paths:
    try:
        # Load image directly with PIL to reduce memory usage
        pil_image = Image.open(img_path).convert('RGB')
        original_size = pil_image.size  # (width, height)

        # Convert PIL image to numpy array
        image = np.array(pil_image)

        # Perform object detection within torch.no_grad() context
        with torch.no_grad():
            results = yolo_model.predict(
                source=image,
                device=device,
                half=True,
                verbose=False,
                conf=0.3 # Set confidence threshold here
            )

        # Create mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            if conf > 0.3:
                mask[y1:y2, x1:x2] = 255

        # Convert mask to PIL image
        pil_mask = Image.fromarray(mask)

        # Pass the original PIL images to SimpleLama within torch.no_grad()
        with torch.no_grad():
            result = simple_lama(pil_image, pil_mask)

        # Save the result
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        result.save(output_path)

        # Free up memory
        del pil_image, pil_mask, result, image, mask, boxes, results
        gc.collect()
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"Error processing {img_path}: {e}")
        gc.collect()
        torch.cuda.empty_cache()
