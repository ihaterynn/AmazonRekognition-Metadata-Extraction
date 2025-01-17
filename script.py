import os
import boto3
import json
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Initialize Amazon Rekognition Client
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

# Specific file names to process
specific_files = [
    "GN BLOOM 82090-4.jpg",
    "GN BLOOM 82090-6.jpg",
    "GN BLOOM 82095-2.jpg",
    "GN CAR 81330-2.jpg",
    "GN CAR 81333-5.jpg",
    "GN CAR 81334-2.jpg",
    "GN CAR 81334-4.jpg", 
    "GN CAR 81338-2.jpg",
    "HERA6 6089-2.jpg",
    "HERA6 6092-3.jpg",
    "HERA6 6094-2.jpg",
    "HERA6 6100-1.jpg",
    "HERA6 6101-1.jpg",
    "HERA6 6101-3.jpg",
    "JOINUS 8606-1.jpg",
    "JOINUS 8608-1.jpg",
    "JOINUS 8612-2.jpg",
    "JOINUS 8617-2.jpg",
    "JOINUS 8617-7.jpg",
    "JOINUS 8621-2.jpg",
    "JOINUS 8625-1.jpg",
    "JOINUS 8626-3.jpg",
    "JOINUS 8627-3.jpg",
    "JOINUS WH501-1.jpg",
    "JOINUS WH503-1.jpg"
]

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
def process_images(folder_path, specific_files, output_file):
    global api_call_count  
    results = []
    for idx, file_name in enumerate(specific_files, start=1):
        image_path = os.path.join(folder_path, file_name)
        if not os.path.exists(image_path):
            print(f"File not found: {file_name}")
            results.append({"file_name": file_name, "error": "File not found"})
            continue

        # Resize image if necessary
        resized_path = resize_image(image_path)
        if resized_path is None:
            print(f"Skipping {file_name}: Could not resize to under 5 MB")
            continue

        print(f"Processing image {idx}/{len(specific_files)}: {file_name}")

        # Read the image as bytes
        with open(resized_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Call Rekognition for label detection and image properties
        try:
            label_response = rekognition.detect_labels(
                Image={'Bytes': image_bytes},
                MaxLabels=10,  # Limit the number of labels returned
                Features=["GENERAL_LABELS", "IMAGE_PROPERTIES"]  # Enable image properties
            )
            api_call_count += 1  # Increment API call count

            # Extract labels and confidence scores
            labels = [
                {"description": label["Name"], "score": label["Confidence"]}
                for label in label_response["Labels"]
            ]

            # Extract dominant colors and image quality metrics
            if "ImageProperties" in label_response:
                image_properties = label_response["ImageProperties"]
                print(f"Image Properties: {json.dumps(image_properties, indent=4)}")  # Debug print

                # Extract dominant colors with CSS color and simplified color
                dominant_colors = [
                    {
                        "HexCode": color.get("HexCode", "N/A"),  # Safe access to 'HexCode'
                        "RGB": f"({color.get('Red', 'N/A')}, {color.get('Green', 'N/A')}, {color.get('Blue', 'N/A')})",  # RGB as a tuple
                        "Percentage": color.get("PixelPercent", "N/A"),  # Safe access to 'PixelPercent'
                        "CSSColor": color.get("CSSColor", "N/A"),  # CSSColor value
                        "SimplifiedColor": color.get("SimplifiedColor", "N/A")  # SimplifiedColor value
                    }
                    for color in image_properties.get("DominantColors", [])
                ]
            else:
                dominant_colors = []
            
            # Extract quality metrics
            quality_metrics = image_properties.get("Quality", {}) if "ImageProperties" in label_response else {}

            # Append results for this image
            results.append({
                "file_name": file_name,
                "labels": labels,
                "dominant_colors": dominant_colors,
                "quality_metrics": quality_metrics
            })

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

# Process Images Using Specific File Names
process_images(input_folder, specific_files, output_file)
