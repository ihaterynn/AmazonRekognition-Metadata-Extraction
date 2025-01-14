import os
import boto3
import json
import random
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Use credentials from .env file
rekognition = boto3.client(
    'rekognition',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

print("Amazon Rekognition API client initialized successfully.")

# Input and Output Paths
input_folder = "./Rezised Wallpapers/"
output_file = "./results/rekognition_output.json"

# API Call Counter
api_call_count = 0  # Track API calls

# Function to Select 50% of Images Randomly
def select_images(folder_path, percentage=0.5):
    all_images = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
    selected_images = random.sample(all_images, int(len(all_images) * percentage))
    return selected_images

def resize_image(image_path, max_size_mb=5):
    max_size_bytes = max_size_mb * 1024 * 1024
    if os.path.getsize(image_path) <= max_size_bytes:
        return image_path  # Skip resizing if the image is already under the size limit

    try:
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ("RGBA", "P"):  # RGBA and P (palette) modes need conversion
                img = img.convert("RGB")

            # Reduce size by scaling down
            img.thumbnail((2000, 2000))  # Adjust dimensions as needed
            temp_path = os.path.join(os.path.dirname(image_path), "temp_" + os.path.basename(image_path))
            img.save(temp_path, optimize=True, quality=85)  # Save with reduced quality to shrink size

            # Check if resized image is below the size limit
            if os.path.getsize(temp_path) <= max_size_bytes:
                print(f"Resized {image_path} to under {max_size_mb} MB")
                return temp_path
            else:
                print(f"Unable to resize {image_path} to under {max_size_mb} MB")
                return None
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None

# Function to Process Images Using Amazon Rekognition
def process_images(folder_path, selected_images, output_file):
    global api_call_count  
    results = []
    for idx, file_name in enumerate(selected_images, start=1):
        image_path = os.path.join(folder_path, file_name)

        # Resize image if necessary
        resized_path = resize_image(image_path)
        if resized_path is None:
            print(f"Skipping {file_name}: Could not resize to under 5 MB")
            continue

        print(f"Processing image {idx}/{len(selected_images)}: {file_name}")

        # Read the image as bytes
        with open(resized_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Call Rekognition for label detection
        try:
            label_response = rekognition.detect_labels(
                Image={'Bytes': image_bytes},
                MaxLabels=10  # Limit the number of labels returned
            )
            api_call_count += 1  # Increment API call count

            # Extract labels and confidence scores
            labels = [
                {"description": label["Name"], "score": label["Confidence"]}
                for label in label_response["Labels"]
            ]

            # Append results for this image
            results.append({"file_name": file_name, "labels": labels})

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results.append({"file_name": file_name, "error": str(e)})

        # Remove temporary resized image
        if resized_path != image_path:
            os.remove(resized_path)

    print(f"Total API calls made: {api_call_count}")

    # Save results to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Processing complete. Results saved to {output_file}")

# Select 50% of Images
selected_images = select_images(input_folder)

# Process Images and Extract Metadata
process_images(input_folder, selected_images, output_file)
