import cv2
import mediapipe as mp
import time
import pyautogui


class Autogui:
  def __init__(self, now_x, now_y, move_rate):
    self.prev_x = -100
    self.prev_y = -100
    self.now_x = now_x
    self.now_y = now_y
    self.move_rate = move_rate

  def moveMause(self, x, y):
    self.prev_x = self.now_x
    self.prev_y = self.now_y
    self.now_x = x
    self.now_y = y

    # 移動量を計算
    mov_x = self.now_x - self.prev_x
    mov_y = self.now_y - self.prev_y

    # 移動量を増やす
    mov_x *= self.move_rate
    mov_y *= self.move_rate
    # 現在座標からmov_x, mov_yだけ移動
    pyautogui.move(mov_x, mov_y)
  
  def click(self):
    pyautogui.click()

def make_landmark_list(image, landmarks):
  print('******************************************************************')
  image_width, image_height = image.shape[1], image.shape[0]
  landmark_list = []

  for _, landmark in enumerate(landmarks.landmark):
    #print(landmark)
    landmark_x = min(int(landmark.x * image_width), image_width - 1)
    landmark_y = min(int(landmark.y * image_height), image_height - 1)
    #$landmark_y = min(int(landmark.x * image_width), image_width - 1)

    landmark_list.append([landmark_x, landmark_y])
  print("000000000000000000000000000000000000000000000000000000000000000")
  return landmark_list

def calc_center(landlist):
  # 掌の大雑把な中心を計算
  center_x = (landlist[2][0] + landlist[17][0]) // 2
  center_y = (landlist[2][1] + landlist[17][1]) // 2
  return center_x, center_y

def main():
  print("Hello")
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  # For static images:
  IMAGE_FILES = []
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print handedness and draw hand landmarks on the image.
      print('Handedness:', results.multi_handedness)
      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
      for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        )
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      cv2.imwrite(
          '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

  # 操作モード
  mode_list = ["Mause", "Zoom", "Keybord"]
  mode = mode_list[0]
  list_idx = 0
  start = 0

  # For webcam input:
  cap = cv2.VideoCapture(0)
  success, image = cap.read()
  # 画像縦横の大きさ
  image_width, image_height = image.shape[1], image.shape[0]
  
  # guiの初期化
  gui = Autogui(image_width, image_height, 2.0)

  with mp_hands.Hands(
      min_detection_confidence=0.7,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        print("Hidetada")
        for hand_landmarks in results.multi_hand_landmarks:
          print("Yamagisi")
          # landmarkをlistに変換
          landmark_list = make_landmark_list(image, hand_landmarks)
          # landmarkから手の位置（重心のようななにか）を計算
          center_x, center_y = calc_center(landmark_list)
          #親指と小指の距離計算
          calc_dist = ((landmark_list[4][0] - landmark_list[20][0] )**2 
                      + (landmark_list[4][1]  - landmark_list[20][1] )**2)**0.5
          
          # モード変更
          if calc_dist < 100:
            if start == 0:
              start = time.time()
            else:
              if time.time() - start > 2:    
                # モード変更
                list_idx += 1
                list_idx %= len(mode_list)
                mode = mode_list[list_idx]
                start = 0

          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
        print("###############################")
        print(mode)
        if mode == mode_list[0]:
          print(center_x, center_y)
          gui.moveMause(center_x, center_y)
          print("ueyama")
          # clickモーションをするか
          click_dist = ((landmark_list[4][0] - landmark_list[16][0])**2
                        + (landmark_list[4][1] - landmark_list[16][1])**2)**0.5
          if click_dist < 100:
            print("rui")
            gui.click()
        elif mode == mode_list[1]:
          pass
        else:
          pass
      cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()


if __name__ == '__main__':
  main()