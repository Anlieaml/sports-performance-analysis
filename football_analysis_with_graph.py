import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to store data for graphing
player_positions = []
frame_numbers = []

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

def draw_tracking_info(frame, contours, frame_num):
    """Draw bounding boxes and track objects."""
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Log player position (center of the bounding box)
            center_x = x + w // 2
            center_y = y + h // 2
            player_positions.append((center_x, center_y))
            frame_numbers.append(frame_num)

def plot_graph():
    """Plot tracked player positions on a graph."""
    x_positions = [pos[0] for pos in player_positions]
    y_positions = [pos[1] for pos in player_positions]

    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, x_positions, label='X Position', color='blue')
    plt.plot(frame_numbers, y_positions, label='Y Position', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Position')
    plt.title('Player Movement Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    video_path = r'C:\Users\DELL\Downloads\video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create a background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error.")
            break

        processed_frame = preprocess_frame(frame)
        contours = detect_objects(processed_frame, background_subtractor)
        draw_tracking_info(frame, contours, frame_num)
        
        frame_num += 1

        # Display the frame
        cv2.imshow("Football Performance Analysis", frame)

        # Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the graph after processing the video
    plot_graph()

if __name__ == "__main__":
    main()