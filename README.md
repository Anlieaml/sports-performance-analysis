sports-performance-analysis
# 🎯 Football Player Tracking and Movement Analysis using OpenCV

This project demonstrates how to use basic computer vision techniques to **track player movement** from a football match video. The code uses **background subtraction** to detect players and logs their position frame by frame. A matplotlib graph is generated at the end to show how the player's position changes over time.

---

## 📌 Features

- Real-time player detection using background subtraction (`cv2.createBackgroundSubtractorMOG2`)
- Bounding boxes around moving players
- Tracking player position (center point of bounding box)
- Logs position across video frames
- Graphical visualization of movement using Matplotlib

---

## 🧰 Requirements

Install the required Python libraries using pip:

```bash
pip install opencv-python numpy matplotlib
📂 File Structure
css
Copy
Edit
📁 project-folder/
├── 📄 player_tracking.py       # Main Python script
├── 🎥 video.mp4                # Input football match video (user-provided)
▶️ How to Run
Place your football match video in the project folder.

Update the video path in the script:

python
Copy
Edit
video_path = r'C:\Users\DELL\Downloads\video.mp4'
Replace with your actual path if needed.

Run the script:

bash
Copy
Edit
python player_tracking.py
Press q to quit the video window anytime.

📊 Output
A live video window showing detected players with green bounding boxes.

After the video ends, a graph is displayed:

Blue line shows X-coordinate movement

Red line shows Y-coordinate movement

X-axis: Frame number

Y-axis: Position in pixels

🧠 How It Works
Preprocessing:

Convert frame to grayscale

Apply Gaussian blur to reduce noise

Detection:

Use background subtraction to isolate moving objects

Apply morphological operations to clean noise

Extract contours (moving players)

Tracking:

For each contour with area > 500, draw a bounding box

Store the center (x, y) of the box for graphing

Visualization:

Use Matplotlib to plot X and Y position over time

🚀 Future Enhancements
Track multiple players with unique IDs

Apply Kalman Filter or SORT for advanced tracking

Export annotated video output

Add speed and distance calculations

🙌 Credits
Developed as part of a Computer Vision project using OpenCV and Python.








