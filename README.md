## ✨ 🅞🅒🅡ina 
###  Hand-Tracked Drawing → OCR → ESP8266 → BCD Hardware Adder


![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Espressif](https://img.shields.io/badge/espressif-E7352C.svg?style=for-the-badge&logo=espressif&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Project Description:

OCRina is a project which combines computer vision, OCR and digital electronics to create a gesture-based BCD calculator.

Using Opencv and Mediapipe, the user writes digits onto a virtual canvas which are then recognised using PyTesseract OCR, the detected numbers are sent to the ESP8266 which output their 2 3-bit binary numbers to a BCD adder hardware circuit built with 7483 adders and 7448 seven-segment decoders.

The system performs real decimal addition and displays the result on physical seven-segment displays.

<div align="center">
  <img src="media/circuit_pic_1.jpeg" height="200">
  <img src="media/circuit_pic_2.jpeg" height="200">
</div>


## Simulated Circuit:
<p align="center">
  <img src="media/multisim_circuit.jpeg" width="400" />
  <br>
  <b>Multisim Circuit</b>
</p>

## [Gesture_Drawing](https://github.com/vedantmalkar/OCRina/blob/main/firmware/main.py) :
This code lets the user draw numbers in the air using hand-tracking, places those strokes onto a virtual canvas, recognizes the digit, and then sends the detected number to the physical BCD adder through the ESP.

<p align="center">
  <img src="https://github.com/user-attachments/assets/35105250-c6ba-4c2a-8f46-93ad395f7ec6" width="550">
</p>

<br>

## [ESP8266_Communication](https://github.com/vedantmalkar/OCRina/blob/main/firmware/esp/esp8266_code.ino):
The ESP8266 code receives the recognized digits over serial communication from the program, converts each digit into its 3-bit binary form, and outputs these bits through the GPIO pins. These signals are then fed directly into the hardware adder circuit for BCD addition

<br>


## Demo Video:
<p align="center">

| <video src="https://github.com/user-attachments/assets/fc4de42c-5be2-44f1-86e5-5cfb62d69c8a" height="100" controls></video> |
|:-----------------------------------------------------------:|
| **Addition of 6 and 7** |

<br>

## Project Collaborators:
- Vedant Malkar
- Zaid Faruqui
- Ananya Rane
- Arnav Shelke
- Mayuresh Surve
