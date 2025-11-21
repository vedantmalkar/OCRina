import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pytesseract

# --- Setup ---
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

# Canvas for drawing
canvas_size = 300
white_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

# Drawing state
drawing_mode = False
cx_prev, cy_prev = 0, 0

# Pointer position on canvas
pointer_pos = None

# FPS
pTime = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Reset pointer by default
    pointer_pos = None

    if results.multi_hand_landmarks:
        for hand_id, handLms in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[hand_id].classification[0].label

            # Collect landmark coordinates
            lm_list = {}
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list[id] = (cx, cy)

            # Draw hand landmarks on camera feed
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # --- Right hand controls drawing ---
            if hand_type == "Right":
                if 8 in lm_list and 4 in lm_list:
                    # Distance between thumb tip and index tip for drawing
                    distance_draw = math.hypot(lm_list[8][0]-lm_list[4][0], lm_list[8][1]-lm_list[4][1])
                    drawing_mode = distance_draw < 40

                    # Map index finger to canvas coordinates
                    cx_draw = int(lm_list[8][0] * canvas_size / w)
                    cy_draw = int(lm_list[8][1] * canvas_size / h)

                    # Update pointer position for display
                    pointer_pos = (cx_draw, cy_draw)

                    # Draw on canvas if in drawing mode
                    if drawing_mode:
                        if cx_prev != 0 and cy_prev != 0:
                            cv2.line(white_canvas, (cx_prev, cy_prev), (cx_draw, cy_draw), (0,0,0), 5)
                        cx_prev, cy_prev = cx_draw, cy_draw
                    else:
                        cx_prev, cy_prev = 0,0

                # Clear canvas gesture (thumb + pinky)
                if 4 in lm_list and 20 in lm_list:
                    distance_clear = math.hypot(lm_list[4][0]-lm_list[20][0], lm_list[4][1]-lm_list[20][1])
                    if distance_clear < 40:
                        white_canvas.fill(255)
                        print("Canvas Cleared!")

            # --- Left hand triggers OCR ---
            if hand_type == "Left":
                if 8 in lm_list and 4 in lm_list:
                    distance_trigger = math.hypot(lm_list[8][0]-lm_list[4][0], lm_list[8][1]-lm_list[4][1])
                    if distance_trigger < 40:
                        # OCR
                        gray_canvas = cv2.cvtColor(white_canvas, cv2.COLOR_BGR2GRAY)
                        _, thresh_canvas = cv2.threshold(gray_canvas, 200, 255, cv2.THRESH_BINARY_INV)
                        text = pytesseract.image_to_string(thresh_canvas, lang="eng")
                        print(f"Detected Text: {text}")

    # --- Display canvas with pointer ---
    display_canvas = white_canvas.copy()
    if pointer_pos:
        cv2.circle(display_canvas, pointer_pos, 7, (0,0,255), -1)  # red pointer

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10,70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)

    # Show
    cv2.imshow("Camera", frame)
    cv2.imshow("Canvas", display_canvas)  # pointer appears only on display

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
