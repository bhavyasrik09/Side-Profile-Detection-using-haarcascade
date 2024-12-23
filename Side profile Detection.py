import cv2
import os

# Load the profile face cascade classifier
cascade_classifier = cv2.CascadeClassifier("C:\\Users\\user\\Desktop\\Edgematrix\\xml files\\haarcascade_profileface.xml")

# Function to detect objects and save detected frames
def detect_objects(frame, cascade_classifier, output_folder, frame_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in objects:
        # Draw rectangles around detected objects
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the frame with detected objects
        cv2.imwrite(os.path.join(output_folder, f'detected_profile_face_{frame_count}.png'), frame)

    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window for display
cv2.namedWindow('Profile Face Detection')

# Create the output folder if it does not exist
output_folder = 'detected_profile_faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0

# Main loop for detecting profile faces
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects and update frame count
    frame = detect_objects(frame, cascade_classifier, output_folder, frame_count)
    frame_count += 1

    # Display the frame
    cv2.imshow('Profile Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
