import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
import pyautogui



import os
import sys

if getattr(sys, "frozen", False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "hand_landmarker.task")

base_options = python.BaseOptions(
    model_asset_path=model_path
)
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

last_action_time = 0
cooldown = 0.5

def draw_landmarks(img, detection_results):
  img_copy = np.copy(img)

  for hand_landmarks,hand_handedness in zip(detection_results.hand_landmarks, detection_results.handedness):
    mp_drawing.draw_landmarks(img_copy, 
                              hand_landmarks,
                              mp_hands.HAND_CONNECTIONS, 
                              mp_drawing_styles.get_default_hand_landmarks_style(),
                              mp_drawing_styles.get_default_hand_connections_style())

    index_tip = hand_landmarks[8].y
    index_pip = hand_landmarks[6].y

    ring_tip = hand_landmarks[16].y
    ring_pip = hand_landmarks[14].y

    hand_label = hand_handedness[0].category_name
    
    if hand_label == "Right":
        if index_tip > index_pip:
            pyautogui.press("volumeup")
        elif ring_tip > ring_pip:
            pyautogui.press("volumedown")
    elif hand_label == "Left":
        # Keep this completely blank or use for different commands so it NEVER changes volume
        pass


        
  return img_copy

def detect_hands(image):
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

  detection_result = detector.detect(img)
  drawn_image = draw_landmarks(img.numpy_view(), detection_result)

  return cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)

cap = cv2.VideoCapture(0)

while cap.isOpened():
  ret, frame = cap.read()
  
  if ret:
    processed_frame = detect_hands(frame)
    final_output = cv2.flip(processed_frame, 1)
    cv2.imshow("Audio Hand Control", final_output)

    

    key = cv2.waitKey(1)
    if key == 27:
      break
  else:
    break

            
cap.release()
cv2.destroyAllWindows()