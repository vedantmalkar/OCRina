import serial
import time

# Open serial connection to the ESP8266
def open_serial_connection(port, baudrate=115200):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        ser.reset_input_buffer()
        print("Serial connection established")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

# Send integers to ESP8266
def send_integers_to_esp8266(ser, num1, num2):
    if ser is None:
        print("Serial connection not established")
        return
    message = f"{num1},{num2}\n"  # Format message as comma-separated
    ser.write(message.encode())  # Send over serial
    print(f"Sent: {message.strip()}")
    time.sleep(0.5)  # Small delay

# Read response from ESP8266
def read_response_from_esp8266(ser):
    if ser is None:
        return None
    response_lines = []
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                response_lines.append(line)
                if line == "OK":  # Stop reading when ESP signals done
                    break
        return response_lines
    except UnicodeDecodeError:
        return ["Error decoding response"]
