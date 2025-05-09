import os
import cv2

# Path to the folder containing videos
video_folder = r" "

# Output directory for extracted face images
output_folder = r" "

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Number of frames to extract per video
frames_to_extract = 30

print("Starting face extraction...")

# Iterate over all files in the video folder
for video_file in os.listdir(video_folder):
    if not video_file.endswith('.avi'):
        continue

    video_path = os.path.join(video_folder, video_file)
    print(f"\nProcessing: {video_file}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 or not cap.isOpened():
        print(f"Error: Could not open {video_file}")
        cap.release()
        continue

    step = max(1, total_frames // frames_to_extract)
    extracted_count = 0
    frame_count = 0

    while cap.isOpened() and extracted_count < frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take the first detected face
            # Crop with margin
            y1 = max(0, y - 25)
            y2 = min(frame.shape[0], y + h + 50)
            face = frame[y1:y2, x:x + w]

            # Resize face image
            face_resized = cv2.resize(face, (224, 224))

            # Save the cropped face image
            image_name = f"{os.path.splitext(video_file)[0]}_frame_{extracted_count+1}.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, face_resized)
            print(f"Saved: {image_name}")

            extracted_count += 1

        frame_count += step

    cap.release()
    print(f"Finished: {video_file}")

print("\n Done extracting faces from all videos.")
