import cv2
import numpy as np

# Function to calculate the absolute difference between frames
def frame_diff(prev_frame, current_frame, next_frame):
    diff1 = cv2.absdiff(next_frame, current_frame)
    diff2 = cv2.absdiff(current_frame, prev_frame)
    return cv2.bitwise_and(diff1, diff2)

# Function to detect violence in a video
def detect_violence(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize previous frames
    prev_frame = cap.read()[1]
    current_frame = cap.read()[1]
    next_frame = cap.read()[1]

    while cap.isOpened():
        # Calculate frame difference
        frame_delta = frame_diff(prev_frame, current_frame, next_frame)

        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        violence_detected = False

        # Check if any contour has a significant area (adjust as needed)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                violence_detected = True
                print("Violence detected in the frame!")
                # You can take further actions or mark the frame as violent

        # Copy the frame to avoid modifying the original
        frame_with_text = next_frame.copy()

        # Draw text on the frame based on violence detection
        if violence_detected:
            cv2.putText(frame_with_text, "Violence Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame_with_text, "No Violence", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame_with_text)

        # Update previous frames
        prev_frame = current_frame.copy()
        current_frame = next_frame.copy()

        # Read the next frame
        ret, next_frame = cap.read()

        if not ret:
            break

        # Introduce a delay (30 milliseconds) to slow down the video
        cv2.waitKey(30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'test_dataset/violence.mp4'
detect_violence(video_path)
