import tkinter as tk
from PIL import Image, ImageTk
import csv
from datetime import date
import time
import numpy as np
import cv2
from tkinter.filedialog import askopenfilename
import os
from tkinter import messagebox as ms
import shutil
from playsound import playsound  # Import the playsound library

global fn

#==============================================================================
root = tk.Tk()
root.state('zoomed')

root.title("Deepfake Detection System")

current_path = str(os.path.dirname(os.path.realpath('__file__')))

basepath = current_path + "\\"

#==============================================================================
#==============================================================================

img = Image.open(basepath + "img1.jpg")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

bg = img.resize((w, h), Image.LANCZOS)

bg_img = ImageTk.PhotoImage(bg)

bg_lbl = tk.Label(root, image=bg_img)
bg_lbl.place(x=0, y=0)

heading = tk.Label(root, text="Deepfake Detection System", width=45, font=("Times New Roman", 45, 'bold'), bg="#192841", fg="white")
heading.place(x=0, y=0)

def show_FDD_video(video_path):
    ''' Display FDD video with annotated bounding box and labels '''
    from tensorflow.keras.models import load_model

    img_cols, img_rows = 64, 64

     

    FALLModel = load_model(r'D:/100% code fake video detection/100% code/fake_event.h5')

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("{} cannot be opened".format(video_path))
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    red = (0, 0, 255)
    line_type = cv2.LINE_AA
    i = 1

    while True:
        ret, frame = video.read()

        if not ret:
            break

        img = cv2.resize(frame, (img_cols, img_rows), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)

        X_img = img.reshape(-1, img_cols, img_rows, 1)
        X_img = X_img.astype('float32')

        X_img /= 255

        predicted = FALLModel.predict(X_img)

        if predicted[0][0] < 0.5:
            predicted[0][0] = 0
            predicted[0][1] = 1
            label = 1
        else:
            predicted[0][0] = 1
            predicted[0][1] = 0
            label = 0

        frame_num = int(i)
        label_text = ""

        color = (255, 255, 255)

        if label == 1:
            label_text = "Fake Image Detected"
            color = red

            # Play a sound when a fake image is detected
            playsound('voice.mp3')  # Replace 'alert_sound.mp3' with the path to your sound file

        else:
            label_text = "Normal Image Detected"
            color = green

        frame = cv2.putText(
            frame, "Frame: {}".format(frame_num), (5, 30),
            fontFace=font, fontScale=1, color=color, lineType=line_type
        )
        frame = cv2.putText(
            frame, "Label: {}".format(label_text), (5, 60),
            fontFace=font, fontScale=1, color=color, lineType=line_type
        )

        i = i + 1
        cv2.imshow('FDD', frame)
        if cv2.waitKey(30) == 27:
            break

    video.release()
    cv2.destroyAllWindows()

def Video_Verify():
    global fn

    fileName = askopenfilename(initialdir='/', title='Select image',
                               filetypes=[("all files", "*.*")])

    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.')[-1]  # Correctly get the file extension


    if Sel_F != 'mp4':
        print("Select Video File!!!!!!")
    else:
        show_FDD_video(fn)

def upload():
    global fn

    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])

    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)

    if Sel_F != 'mp4':
        print("Select Video File!!!!!!")
        ms.showerror('Oops!', 'Select Video File!!!!!!')
    else:
        ms.showinfo('Success!', 'Video Uploaded Successfully !')
        return fn

def convert():
    cam = cv2.VideoCapture(fn)
    try:
        if not os.path.exists('images'):
            os.makedirs('images')
    except OSError:
        print('Error: Creating directory of images')

    currentframe = 0

    while True:
        ret, frame = cam.read()

        if ret:
            name = './images/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            cv2.imwrite(name, frame)

            currentframe += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()
    ms.showinfo('Success!', 'Video converted into frames Successfully !')

def CLOSE():
    root.destroy()

button2 = tk.Button(root, command=Video_Verify, text="Detect Fake Video", width=20, font=("Times new roman", 25, "bold"), bg="cyan", fg="black", bd=5, relief="ridge")
button2.place(x=100, y=200)

close = tk.Button(root, command=CLOSE, text="Exit", width=20, font=("Times new roman", 25, "bold"), bg="red", fg="white", bd=5, relief="ridge")
close.place(x=100, y=350)

root.mainloop()