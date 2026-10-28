'''When this code was created only us and God knew how it worked. 
Now only God knows. I would suggest you dont try messing with it'''
import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
import pyautogui
import time
pyautogui.PAUSE = 0


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



model_path_gest = os.path.join(base_dir, "gesture_recognizer.task")

base_options_gest = python.BaseOptions(
    model_asset_path=model_path_gest
)
options_gest = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options_gest, num_hands=2)
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options_gest)

mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

last_action_time = 0
cooldown = 0.3
fist_active = False

def draw_landmarks(img, detection_results, gesture_results):
  global fist_active
  global last_action_time
  img_copy = np.copy(img)

  for i,(hand_landmarks,hand_handedness) in enumerate(zip(detection_results.hand_landmarks, detection_results.handedness)):
    mp_drawing.draw_landmarks(img_copy, 
                              hand_landmarks,
                              mp_hands.HAND_CONNECTIONS, 
                              mp_drawing_styles.get_default_hand_landmarks_style(),
                              mp_drawing_styles.get_default_hand_connections_style())

    index_y = hand_landmarks[8].y
    index_x = hand_landmarks[8].x

    thumb_y = hand_landmarks[4].y
    thumb_x = hand_landmarks[4].x

    middle_y = hand_landmarks[12].y
    middle_x = hand_landmarks[12].x

    ring_y = hand_landmarks[16].y
    ring_x = hand_landmarks[16].x

    pinky_y = hand_landmarks[20].y
    pinky_x = hand_landmarks[20].x

    gest = None
    if i < len(gesture_results.gestures) and gesture_results.gestures[i]:
        gest = gesture_results.gestures[i][0].category_name


    dist_vup = np.sqrt((index_x-thumb_x)**2 + (index_y-thumb_y)**2)
    dist_vdown = np.sqrt((middle_x-thumb_x)**2 + (middle_y-thumb_y)**2)
    dist_mnext = np.sqrt((ring_x-thumb_x)**2 + (ring_y-thumb_y)**2)
    dist_mprev = np.sqrt((pinky_x-thumb_x)**2 + (pinky_y-thumb_y)**2)

    hand_label = hand_handedness[0].category_name
    
    if hand_label == "Right":
        if gest == "Closed_Fist":
            if not fist_active:
                pyautogui.press("playpause")
            fist_active = True
        else:
          fist_active = False
          if time.time() - last_action_time > cooldown:
              if dist_vup < 0.035 and dist_vdown > 0.035:
                  pyautogui.press("volumeup")
                  last_action_time = time.time()
              elif dist_vdown < 0.035 and dist_vup > 0.035:
                  pyautogui.press("volumedown")
                  last_action_time = time.time()
              elif dist_mnext < 0.035:
                  pyautogui.press('nexttrack')
                  last_action_time = time.time()
              elif dist_mprev < 0.035:
                  pyautogui.press('prevtrack')  
                  last_action_time = time.time()
              elif dist_vdown < 0.035 and dist_vup < 0.035:
                 pyautogui.press("volumemute")
                 last_action_time = time.time()
        
    elif hand_label == "Left":
        # Keep this completely blank or use for different commands so it NEVER changes volume
        pass

  return img_copy

def detect_hands(image):
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

  detection_result = detector.detect(img)
  gesture_result = recognizer.recognize(img)   # clean image, once per frame
  drawn_image = draw_landmarks(img.numpy_view(), detection_result, gesture_result)

  return cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)

cap = cv2.VideoCapture(0)
window_name = "Media Playback Hand Controller"
while cap.isOpened():
  ret, frame = cap.read()
  
  if ret:
    processed_frame = detect_hands(frame)
    final_output = cv2.flip(processed_frame, 1)
    cv2.imshow(window_name, final_output)

    cv2.waitKey(1)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
      break
  else:
    break

            
cap.release()
cv2.destroyAllWindows()