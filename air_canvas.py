import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Color Options (BGR Format)
colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}
draw_color = "red"
brush_thickness = 5
selection_mode = False  # Prevents accidental drawing after selection

# Screen Size (HD 1280x720)
WIDTH, HEIGHT = 1280, 720

# UI Positions
color_palette = [
    ((20, 20), (120, 100), "red"),
    ((140, 20), (240, 100), "green"),
    ((260, 20), (360, 100), "blue"),
    ((380, 20), (480, 100), "yellow"),
    ((500, 20), (600, 100), "eraser"),
]

brush_size_ui = [
    ((640, 20), (740, 100), 5),
    ((760, 20), (860, 100), 10),
    ((880, 20), (980, 100), 15),
]

# Store Drawn Points
def reset_canvas():
    """ Clears all drawing points. """
    global points
    points = {color: deque(maxlen=2048) for color in colors}

reset_canvas()
prev_point = None  # Track previous point

# Open Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Reduce width to 640
cap.set(4, 480)  # Reduce height to 480


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw UI Elements
    for (top_left, bottom_right, color) in color_palette:
        cv2.rectangle(frame, top_left, bottom_right, (colors[color] if color != "eraser" else (0, 0, 0)), -1)
        cv2.putText(frame, color.capitalize(), (top_left[0] + 10, top_left[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for (top_left, bottom_right, size) in brush_size_ui:
        cv2.rectangle(frame, top_left, bottom_right, (200, 200, 200), -1)
        cv2.putText(frame, str(size), (top_left[0] + 20, top_left[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Ensure index finger is valid
            if landmarks and len(landmarks) > 8:
                index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)

                # Check Fingers Up
                index_up = landmarks[8].y < landmarks[6].y
                middle_up = landmarks[12].y < landmarks[10].y

                # If Selecting, Disable Drawing
                if selection_mode:
                    if not index_up:  # Exit selection mode when the finger is lifted
                        selection_mode = False
                    continue

                # If Only Index Finger Up → Drawing Mode
                if index_up and not middle_up and not selection_mode:
                    if prev_point is not None:
                        points[draw_color].append((index_x, index_y))
                    prev_point = (index_x, index_y)

                # Check for Color or Eraser Selection
                for (top_left, bottom_right, color) in color_palette:
                    if top_left[0] < index_x < bottom_right[0] and top_left[1] < index_y < bottom_right[1]:
                        if color == "eraser":
                            reset_canvas()  # ✅ Clears Screen
                        else:
                            draw_color = color
                        prev_point = None  
                        selection_mode = True  
                        break

                # Check for Brush Size Selection
                for (top_left, bottom_right, size) in brush_size_ui:
                    if top_left[0] < index_x < bottom_right[0] and top_left[1] < index_y < bottom_right[1]:
                        brush_thickness = size
                        prev_point = None  
                        selection_mode = True  
                        break

            # Draw Hand Landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw Points on Canvas
    for color, pts in points.items():
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(frame, pts[i - 1], pts[i], colors[color], brush_thickness)

    # Display Canvas
    cv2.imshow("AI Air Canvas (Fixed)", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
