import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Variables for velocity tracking ---
prev_centroid = None
prev_time = None
prev_angle = None

# Exponential smoothing factors
alpha_vel = 0.3
alpha_ang = 0.3

# Smoothed outputs
smoothed_velocity = np.array([0.0, 0.0, 0.0])
smoothed_angular_velocity = 0.0

# Webcam capture
cap = cv2.VideoCapture(0)

def compute_hand_centroid(landmarks):
    # Compute average position of wrist + MCP joints for robustness"""
    indices = [0, 5, 9, 13, 17]  # wrist + knuckles
    pts = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])
    return np.mean(pts, axis=0)

def compute_palm_axis_angle(landmarks):
    # Compute the angle of the palm axis (index MCP → pinky MCP) in radians"""
    v = np.array([landmarks[5].x - landmarks[17].x,
                  landmarks[5].y - landmarks[17].y])
    return math.atan2(v[1], v[0])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        # Compute centroid
        centroid = compute_hand_centroid(lm)

        # Compute velocity and apply EMA smoothing
        curr_time = time.time()
        if prev_centroid is not None:
            dt = curr_time - prev_time
            velocity = (centroid - prev_centroid) / dt
            smoothed_velocity = alpha_vel * velocity + (1 - alpha_vel) * smoothed_velocity

            print(f"Velocity: {smoothed_velocity}")

        prev_centroid = centroid
        prev_time = curr_time

        # Compute palm axis angle and angular velocity
        angle = compute_palm_axis_angle(lm)
        if prev_angle is not None:
            angular_velocity = (angle - prev_angle) / dt
            smoothed_angular_velocity = alpha_ang * angular_velocity + (1 - alpha_ang) * smoothed_angular_velocity
            print(f"Angular Velocity (deg/s): {math.degrees(smoothed_angular_velocity):.2f}")

        prev_angle = angle

        # Draw landmarks on frame
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Landmark Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()