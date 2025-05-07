import cv2
import numpy as np

def preprocess_frame(frame):
    """Convert frame to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_objects(frame, background_subtractor):
    """Detect moving objects using background subtraction."""
    mask = background_subtractor.apply(frame)
    _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_tracking_info(frame, contours):
    """Draw bounding boxes and track objects."""
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main():
    video_path = "football_match.mp4"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create a background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error.")
            break

        processed_frame = preprocess_frame(frame)
        contours = detect_objects(processed_frame, background_subtractor)
        draw_tracking_info(frame, contours)

        # Display the frame
        cv2.imshow("Football Performance Analysis", frame)

        # Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()