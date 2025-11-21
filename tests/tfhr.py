import cv2
import mediapipe as mp
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load pre-trained MNIST model
model = keras.models.load_model('mnist_model.h5')

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

drawing_mode = False
cx_8_drawing_previous, cy_8_drawing_previous = 0, 0
white_canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255

cx_8, cy_8 = 0, 0
cx_4, cy_4 = 0, 0
cx_20, cy_20 = 0, 0

numbers_drawn = []
pTime = 0

def preprocess_canvas(canvas):
    """Preprocess canvas for MNIST model (28x28, normalized)"""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Invert: white bg becomes 0, black lines become 255
    gray = 255 - gray
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))
    # Normalize to 0-1
    normalized = resized.astype('float32') / 255.0
    return normalized

def predict_digit(canvas):
    """Predict digit from canvas"""
    img = preprocess_canvas(canvas)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence

while True:
    success, flipped = cap.read()
    img = cv2.flip(flipped, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if results.multi_hand_landmarks:
        for hand_id, handLms in enumerate(results.multi_hand_landmarks):
            h, w, c = img.shape
            h_canva, w_canva, c_canva = white_canvas.shape
            
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=2)
                
                if hand_id == 0:
                    if id == 8:
                        cx_8, cy_8 = cx, cy
                    elif id == 4:
                        cx_4, cy_4 = cx, cy
                    elif id == 20:
                        cx_20, cy_20 = cx, cy
                
                if hand_id == 1 and id == 8:
                    cx_8_drawing = int(lm.x * w_canva)
                    cy_8_drawing = int(lm.y * h_canva)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            if hand_id == 0:
                distance_draw = math.sqrt((cx_4 - cx_8)**2 + (cy_4 - cy_8)**2)
                distance_clear = math.sqrt((cx_4 - cx_20)**2 + (cy_4 - cy_20)**2)

                if distance_draw < 30:
                    cv2.line(img, (cx_8, cy_8), (cx_4, cy_4), (255, 0, 0), 5)
                    drawing_mode = True
                else: 
                    drawing_mode = False
                
                # Pinky + thumb = recognize digit
                if distance_clear < 40:
                    digit, confidence = predict_digit(white_canvas)
                    if confidence > 0.7:  # Only accept high confidence predictions
                        numbers_drawn.append(digit)
                        print(f"Detected: {digit} (confidence: {confidence:.2f})")
                        print(f"Numbers so far: {numbers_drawn}")
                        
                        # If we have 2 numbers, send to ESP8266
                        if len(numbers_drawn) == 2:
                            sum_result = numbers_drawn[0] + numbers_drawn[1]
                            print(f"Sum: {numbers_drawn[0]} + {numbers_drawn[1]} = {sum_result}")
                            # TODO: Send sum_result to ESP8266
                            numbers_drawn = []
                    
                    # Clear canvas
                    white_canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255

            if hand_id == 1 and drawing_mode:
                cx_8_drawing = int(results.multi_hand_landmarks[1].landmark[8].x * w_canva)
                cy_8_drawing = int(results.multi_hand_landmarks[1].landmark[8].y * h_canva)
                
                if cx_8_drawing_previous == 0 and cy_8_drawing_previous == 0:
                    cx_8_drawing_previous, cy_8_drawing_previous = cx_8_drawing, cy_8_drawing
                else:
                    cv2.line(white_canvas, (cx_8_drawing_previous, cy_8_drawing_previous), 
                            (cx_8_drawing, cy_8_drawing), (0, 0, 0), 5)
                    cx_8_drawing_previous, cy_8_drawing_previous = cx_8_drawing, cy_8_drawing

            elif not drawing_mode:
                cx_8_drawing_previous = 0
                cy_8_drawing_previous = 0
    
    cTime = time.time()
    fps = 1/(cTime - pTime) if pTime else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
    cv2.putText(img, f"Numbers: {numbers_drawn}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), thickness=1)
    
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", white_canvas)

cap.release()
cv2.destroyAllWindows()