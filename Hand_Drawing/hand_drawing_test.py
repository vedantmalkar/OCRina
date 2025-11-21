import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pytesseract

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()  #(self,static_image_mode,max_num_hands,min_detection_confidence,min_tracking_confidence) - default: (False,2,0.5,0.5)
mpDraw = mp.solutions.drawing_utils 

frame_interval = 100
frame_count = 0
pTime = 0   #previous time
cTime = 0   #current time
drawing_mode = False
cx_8_drawing_previous,cy_8_drawing_previous = 0,0
white_canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255

while True:
    success, flipped = cap.read()
    img = cv2.flip(flipped,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) 
    # print(results.multi_hand_landmarks)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if results.multi_hand_landmarks:
        for hand_id,handLms in enumerate(results.multi_hand_landmarks):
            for id, lm in enumerate(handLms.landmark):
                # print(f'{id}: {lm}')
                h,w,c = img.shape
                h_canva,w_canva,c_canva = white_canvas.shape
                cx,cy = int(lm.x * w) , int(lm.y * h)
                cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),thickness=2)
                
                if hand_id == 0 and id == 8:
                    cx_8 = cx
                    cy_8 = cy
                
                if hand_id == 0 and id == 4:
                    cx_4 = cx
                    cy_4 = cy

                if hand_id == 0 and id == 20:
                    cx_20 = cx
                    cy_20 = cy
                
                if hand_id == 1 and id == 8:
                    cx_8_drawing = int(lm.x * w_canva)
                    cy_8_drawing = int(lm.y * h_canva)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            distance_draw = math.sqrt((cx_4 - cx_8)**2 + (cy_4 - cy_8)**2)
            distance_clear = math.sqrt((cx_4 - cx_20)**2 + (cy_4 - cy_20)**2)

            if distance_draw < 30 :
                cv2.line(img,(cx_8,cy_8),(cx_4,cy_4),(255,0,0) , 5)
                drawing_mode = True
            else: 
                drawing_mode = False
            
            # if distance_clear < 40:
            #     white_canvas.fill(255)

                if distance_clear < 40:
                    text = pytesseract.image_to_string(white_canvas, lang="eng")
                    print(f"Detected Text: {text}")

            if hand_id == 1 and drawing_mode:
                if cx_8_drawing_previous == 0 and cy_8_drawing_previous == 0:
                    cv2.line(white_canvas,(cx_8_drawing_previous,cy_8_drawing_previous),(cx_8_drawing,cy_8_drawing),(255,255,255) , 5)
                    cx_8_drawing_previous,cy_8_drawing_previous = cx_8_drawing,cy_8_drawing
                else:
                    cv2.line(white_canvas,(cx_8_drawing_previous,cy_8_drawing_previous),(cx_8_drawing,cy_8_drawing),(0,0,0) , 5)
                    #time.sleep(0.1)
                    cx_8_drawing_previous,cy_8_drawing_previous = cx_8_drawing,cy_8_drawing

            elif not drawing_mode:
                cx_8_drawing_previous = 0
                cy_8_drawing_previous = 0
    frame_count += 1
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),thickness=10)  
    cv2.imshow("Image",img)
    cv2.imshow("Canva", white_canvas)
