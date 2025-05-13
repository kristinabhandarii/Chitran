import pandas as pd
import numpy as np
import os
import random
from colorthief import ColorThief
from PIL import Image, ImageDraw, ImageFont
import textwrap
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import io
from ultralytics import YOLO

model = models.densenet201(pretrained=True)
model.eval()
yolo_model = YOLO("yolov8s.pt")

def preprocess_image(image):
    original_img = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(original_img).unsqueeze(0).to("cpu")
    return original_img, img_tensor

def get_object_detections(image):
    """Detect objects in the image using YOLO and return a binary mask."""
    model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano model (or use yolov8s.pt)

    results = model(image)  # Run object detection
    mask = np.zeros((image.height, image.width), dtype=np.uint8)  # Create empty mask

    for result in results:
        for box in result.boxes.xyxy:  # Extract bounding boxes
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 255  # Mark detected objects in the mask

    return mask  

def generate_saliency_map(model, img_tensor, original_img):
    """Generate saliency map using DenseNet201"""
    # original_img, img_tensor = preprocess_image(image_path)
    img_tensor.requires_grad_()
    output = model(img_tensor)
    score, _ = output.max(1)
    score.backward()
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = cv2.resize(saliency, (original_img.width, original_img.height))

    # Apply the object detection mask (i.e., exclude detected objects from the saliency map)
    object_mask = get_object_detections(original_img)
    saliency[object_mask == 255] = 0  # Set the detected object areas' saliency to 0

    return saliency

def dynamic_thresholding(saliency_map, percentile=40):
    """Dynamically determine the threshold based on the saliency map"""
    threshold_value = np.percentile(saliency_map, percentile)
    print(f"Dynamic threshold set to: {threshold_value}")
    _, binary_map = cv2.threshold(saliency_map, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_map

def find_coordinates(original_img, saliency_map, threshold=0.05, min_area_threshold=10000):
    # Resize the saliency map to match the dimensions of the original image
    original_img_np = np.array(original_img)  

    # Ensure the image is in the correct format (grayscale or single channel)
    width, height = original_img.size
    
    saliency_resized = cv2.resize((saliency_map * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

    # Apply threshold to get high-saliency areas
    _, binary_map = cv2.threshold(saliency_resized, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Find contours of high-saliency regions
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print("No high-saliency regions detected.")
        return np.array(original_img)

    # Find the contour with the largest number of bright pixels
    best_contour = None
    max_bright_pixels = 0

    for contour in contours:
        # Create a mask for the current contour
        mask = np.zeros_like(binary_map)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Count the number of bright pixels within the contour
        bright_pixels = cv2.countNonZero(cv2.bitwise_and(binary_map, binary_map, mask=mask))

        # Update if this contour has more bright pixels
        if bright_pixels > max_bright_pixels:
            max_bright_pixels = bright_pixels
            best_contour = contour

    if best_contour is None:
        print("No valid salient region found.")
        return np.array(original_img)

    # Get the bounding rectangle for the densest salient region
    x, y, w, h = cv2.boundingRect(best_contour)
    
    if w * h > min_area_threshold:
        return x, y, w, h

    if w * h < min_area_threshold:
        return None, None, None, None


def generate_text_colour(image, coordinates):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Save as PNG or JPEG
    img_byte_arr.seek(0)  # Reset file pointer to the start

    color_thief = ColorThief(img_byte_arr)
    palette = color_thief.get_palette(color_count=6)

    x, y, w, h = coordinates
    cropped_image = image.crop((x, y, (x + w), (y + h)))

    img_byte_arr = io.BytesIO()
    cropped_image.save(img_byte_arr, format='PNG')  # Save as PNG or JPEG
    img_byte_arr.seek(0)  # Reset file pointer to the start
    
    color_thief = ColorThief(img_byte_arr)
    dominant_color = color_thief.get_color(quality=1)
    
    
    def rgb_distance(color1, color2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

    # Largest Euclidean distance
    max_distance = 0
    best_contrast_color = None
    black = (0, 0, 0)
    white = (255, 255, 255)

    def luminance(color):
        """Calculate the relative luminance of an RGB color."""
        r, g, b = [c / 255.0 for c in color]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for color in palette:
        dist = rgb_distance(dominant_color, color)
        if dist > max_distance:
            max_distance = dist
            text_color = color
        if max_distance<70:
            text_color = white if luminance(dominant_color)<0.5 else black
    
    return dominant_color, text_color


def generate_font_style():
    fonts_path = "models/fonts"
    font_files = [os.path.join(fonts_path, font) for font in os.listdir(fonts_path) if font.endswith(('.ttf', '.otf', '.TTF'))]
    selected_font = random.choice(font_files)
    return selected_font


def calculate_max_font_size(image, text, coordinates, font_path, padding=10):
    x,y,w,h = coordinates

    max_font_size = 50 # Start with a large font size
    min_font_size = 10   # Define a minimum font size to avoid infinite loops
    best_font_size = min_font_size
    best_wrapped_lines = []

    while max_font_size >= min_font_size:
        # Try the current font size
        current_font_size = (max_font_size + min_font_size) // 2
        font = ImageFont.truetype(font_path, current_font_size)

        # Wrap text based on the width of the bounding box
        draw = ImageDraw.Draw(image)

        parts = text.split("!", maxsplit=1)
        first_part = parts[0] + "!"
        second_part = parts[1].strip() if len(parts)>1 else ""

        wrapped_lines = []
        for part in [first_part, second_part]:
            if part:  # Process each part if it exists
                words = part.split(" ")
                line = ""
                for word in words:
                    test_line = line + word + " "
                    line_width = draw.textlength(test_line, font=font)  # Use the font object here
                    if line_width <= w:
                        line = test_line
                    else:
                        wrapped_lines.append(line.strip())
                        line = word + " "
                if line:
                    wrapped_lines.append(line.strip())

        # Calculate total height of the text with the current font size
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
        line_spacing = int(line_height * 0.5)
        total_text_height = len(wrapped_lines) * (line_height + line_spacing) - line_spacing + 2*padding

        # Check if the total text height fits within the height of the bounding box
        if total_text_height <= h:
            best_font_size = current_font_size  # Update best font size
            best_wrapped_lines = wrapped_lines[:]
            min_font_size = current_font_size + 1  # Try larger sizes
        else:
            max_font_size = current_font_size - 1  # Try smaller sizes

    return best_font_size, wrapped_lines


def add_transparent_box(image, coordinates, color, transparency):
    x, y, w, h = coordinates
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    box_color = color + (transparency,)
    draw.rectangle([x, y, x + w, y + h], fill=box_color)

def generate_styled_image(text, image, festival):
    image_copy = image.copy()

    common_size = (700, 700)
    image_copy = image_copy.resize(common_size)

    original_img, img_tensor = preprocess_image(image_copy)
    saliency = generate_saliency_map(model, img_tensor, image)
    x, y, w, h = find_coordinates(original_img, saliency, threshold=0.02)
    coordinates = x, y, w, h
    
    font_style = generate_font_style()
    
    
    if (festival and festival.lower() in ["holi", "dashain"]) or (coordinates == (None, None, None, None)):
        font_size = 25
        coordinates = 0, 0, image_copy.width, image_copy.height
        x, y, w, h = coordinates
        
        dominant_color, text_color = generate_text_colour(image_copy, coordinates)
        
        font = ImageFont.truetype(font_style, font_size)
    
        parts = text.split("!", maxsplit=1)
        first_part = parts[0] + "!"
        second_part = parts[1].strip() if len(parts)>1 else ""

        draw = ImageDraw.Draw(image_copy)
        
        wrapped_lines = []
        for part in [first_part, second_part]:
            if part:  # Process each part if it exists
                words = part.split(" ")
                line = ""
                for word in words:
                    test_line = line + word + " "
                    line_width = draw.textlength(test_line, font=font)  # Use the font object here
                    if line_width <= image.width:
                        line = test_line
                    else:
                        wrapped_lines.append(line.strip())
                        line = word + " "
                if line:
                    wrapped_lines.append(line.strip())

        # Calculate total height of the text with the current font size
        padding = 20
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
        line_spacing = int(line_height * 0.5)
        total_text_height = len(wrapped_lines) * (line_height + line_spacing) - line_spacing + 3*padding

        blank_space_height = total_text_height  # You can adjust this value
        new_img = np.zeros((original_img.size[0] + blank_space_height, original_img.size[1], 3), dtype=np.uint8)

        # Place the original image below the blank space
        new_img[:blank_space_height] = dominant_color 
        new_img[blank_space_height:] = image_copy
        text_image = new_img
        # return new_img
        
    else:
        dominant_color, text_color = generate_text_colour(image_copy, coordinates)
        padding=10
        font_size, wrapped_lines = calculate_max_font_size(image_copy, text, coordinates, font_style, padding)
        text_image = image_copy
        font = ImageFont.truetype(font_style, font_size)
        alpha_value = 150
        text_image = image_copy
        
    if isinstance(text_image, np.ndarray):
        text_image = Image.fromarray(text_image)
    draw = ImageDraw.Draw(text_image)

    # Calculate the total text height for vertical centering
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    line_spacing = int(line_height * 0.5)
    total_text_height = len(wrapped_lines) * (line_height + line_spacing) - line_spacing
    # Draw the first part of the text
    y_offset = y + padding  # starting y position, adjust as needed

    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)  # Get bounding box for text
        text_width = bbox[2] - bbox[0]  # width

        x_offset = x + (w - text_width) // 2  # Center the text horizontally
        draw.text((x_offset, y_offset), line, font=font, fill=text_color)
        y_offset += line_height + line_spacing  # Move the y_offset down by the height of the text
    
    text_image = text_image.convert("RGBA")
    
    if image.size != text_image.size:
        text_image = text_image.resize(image.size)
    if image.mode != text_image.mode:
        text_image = text_image.convert(image.mode)
        
    return text_image