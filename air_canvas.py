# air_canvas.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

def run_canvas():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    drawings = deque(maxlen=4096)
    prev_point = None

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    color_index = 0
    draw_color = colors[color_index]

    def fingers_up(hand_landmarks):
        tips_ids = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in tips_ids[1:]:
            fingers.append(1 if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y else 0)
        return fingers

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            h, w, _ = frame.shape

            hand_states = []

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    finger_state = fingers_up(hand_landmarks)
                    hand_states.append(finger_state)

                    index_finger = hand_landmarks.landmark[8]
                    x, y = int(index_finger.x * w), int(index_finger.y * h)

                    if finger_state == [0, 1, 0, 0, 0]:
                        if prev_point is not None:
                            drawings.append((prev_point, (x, y), draw_color))
                        prev_point = (x, y)
                    else:
                        prev_point = None

                    if finger_state == [0, 1, 1, 0, 0]:
                        color_index = (color_index + 1) % len(colors)
                        draw_color = colors[color_index]
                        print(f"Color changed to: {draw_color}")

                    if finger_state == [1, 1, 1, 1, 1]:
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        drawings.clear()
                        print("Canvas cleared!")

            # Save if LEFT hand only shows 5 fingers
            if len(hand_states) == 1 and hand_states[0] == [1, 1, 1, 1, 1]:
                cv2.imwrite("drawing.png", canvas)
                print("✅ Saved drawing as drawing.png")

            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            for pt1, pt2, color in drawings:
                if pt1 is not None and pt2 is not None:
                    cv2.line(canvas, pt1, pt2, color, 5)

            combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
            cv2.rectangle(combined, (10, 10), (60, 60), draw_color, -1)
            cv2.putText(combined, "Press 'q' to quit", (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("🎨 AI Air Canvas (Gesture Controlled)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
