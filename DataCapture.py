import serial
import time
import cv2
import os
import uuid
import threading


current_PWM = ""

def read_serial():
# Create a serial connection
    ser = serial.Serial('COM3', 9600, timeout=1)
    global current_PWM
    # Wait for serial connection to open
    while not ser.is_open:
        time.sleep(0.01)

    try:
        while True:
            if ser.inWaiting() > 0:
                data = ser.readline().decode('utf-8').rstrip()
                current_PWM = data
    finally:
        ser.close()

def count_files_in_dir(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        count += len(files)
    return count


if __name__ == '__main__':

    thread = threading.Thread(target=read_serial)
    thread.start()

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    desired_width = 640
    desired_height = 480

    # Set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    head_folder = 'data'

    img_folder = "img_data"
    pwm_folder = "pwm_data"

    img_folder = os.path.join(head_folder, img_folder)
    pwm_folder = os.path.join(head_folder, pwm_folder)

    if not os.path.exists(img_folder):
        # Create the directory
        os.makedirs(img_folder)
        os.makedirs(pwm_folder)

    prev_time = 0
    fps = 8  # Desired frame rate
    index = int(count_files_in_dir(img_folder))

    if index > 20:
        index = index - 20

    print("sleeping")
    time.sleep(10)

    while True:
        curr_time = time.time()

        # If time elapsed is more than 1/fps, then capture image and read PWM signal
        if curr_time - prev_time > 1./fps:
            prev_time = curr_time
            index = index+1

            # Read the current frame from webcam
            ret, frame = cap.read()

            # Save the image
            img_name = f"{img_folder}/{index}.jpg"
            pwm_name = f"{pwm_folder}/{index}.txt"
            cv2.imwrite(img_name, cv2.resize(frame, (640, 480)))
        
            with open(pwm_name, 'w') as f:
                f.write(f'{current_PWM}')
                print(current_PWM)
