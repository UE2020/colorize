import os
import cv2

count = 0

# Function to extract the first frame from an MP4 file
def extract_first_frame(video_path, output_dir):
    global count
    try:
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        if success:
            # Construct the output file path
            file_name = os.path.basename(video_path)
            frame_output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}-{count}.jpg")
            resized = cv2.resize(frame, (256, 256))
            # Save the first frame as an image
            cv2.imwrite(frame_output_path, resized)
            print(f"Extracted the first frame from {video_path}")
            count += 1
        cap.release()
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")

# Directory containing .mp4 files
input_directory = "/home/tt/Downloads/trailer/trailer"
# Directory where extracted frames will be saved
output_directory = "/home/tt/Downloads/trailer/frames"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Crawl the input directory for .mp4 files
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            extract_first_frame(video_path, output_directory)

print("Extraction complete.")
