import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Initialize variables for counting touches
right_leg_touches = 0
left_leg_touches = 0
ball_rotation = 'unknown'
prev_ball_center = None

# Function to calculate velocity
def calculate_velocity(points, fps):
    if len(points) < 2:
        return 0
    dx = points[-1][0] - points[-2][0]
    dy = points[-1][1] - points[-2][1]
    distance = np.sqrt(dx**2 + dy**2)
    velocity = distance * fps
    return velocity

# List to store player positions
player_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize velocity for the current frame
    velocity = 0

    # YOLO object detection
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    right_ankle_pos = (0, 0)
    left_ankle_pos = (0, 0)

    ball_center = None

    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if class_id == 0:  # class_id 0 is 'person'
            # Extract player bounding box
            player_roi = frame[y1:y2, x1:x2]

            # Draw player bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Pose estimation
            player_pose = pose.process(cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB))
            if player_pose.pose_landmarks:
                # Extract ankle positions
                right_ankle = player_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_ankle = player_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

                right_ankle_pos = (int(right_ankle.x * player_roi.shape[1]) + x1, int(right_ankle.y * player_roi.shape[0]) + y1)
                left_ankle_pos = (int(left_ankle.x * player_roi.shape[1]) + x1, int(left_ankle.y * player_roi.shape[0]) + y1)

                # Draw ankle positions
                cv2.circle(frame, right_ankle_pos, 5, (0, 255, 0), -1)
                cv2.circle(frame, left_ankle_pos, 5, (0, 0, 255), -1)

                # Update player positions for velocity calculation
                player_positions.append((right_ankle_pos[0], right_ankle_pos[1]))
                velocity = calculate_velocity(player_positions, fps)

        elif class_id == 32:  # class_id 32 is 'ball'
            # Extract ball bounding box
            bx1, by1, bx2, by2 = x1, y1, x2, y2

            # Draw ball bounding box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

            # Calculate ball center
            ball_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

            # Calculate ball size
            ball_size = np.sqrt((bx2 - bx1) ** 2 + (by2 - by1) ** 2)

            # Adjust the proximity radius based on the ball size
            proximity_radius = ball_size * 0.6  # Increased factor for proximity

            # Check for interactions with ankles (touch detection)
            def is_within_proximity(ankle_pos, ball_center, radius):
                ax, ay = ankle_pos
                cx, cy = ball_center
                distance = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
                return distance <= radius

            if is_within_proximity(right_ankle_pos, ball_center, proximity_radius):
                right_leg_touches += 1
            if is_within_proximity(left_ankle_pos, ball_center, proximity_radius):
                left_leg_touches += 1

    # Determine ball rotation based on movement direction
    if prev_ball_center is not None and ball_center is not None:
        dx = ball_center[0] - prev_ball_center[0]

        if dx > 0:  # Ball moving to the right
            ball_rotation = 'forward'
        elif dx < 0:  # Ball moving to the left
            ball_rotation = 'backward'

    prev_ball_center = ball_center

    # Overlay annotations
    cv2.putText(frame, f'Right Leg Touches: {right_leg_touches}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Left Leg Touches: {left_leg_touches}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Ball Rotation: {ball_rotation}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Player Velocity: {velocity:.2f} px/s', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display frame with annotations
    cv2.imshow('Annotated Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
