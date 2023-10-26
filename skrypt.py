from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((250, 250))
        img = ImageTk.PhotoImage(image)
        panel1.config(image=img)
        panel1.image = img
        panel1.file_path = file_path

def hide_data():
    data = data_entry.get()
    if hasattr(panel1, 'file_path'):
        image = cv2.imread(panel1.file_path)
        
        # LSB algorithm
        binary_data = ''.join(format(ord(i), '08b') for i in data)
        data_len = len(binary_data)
        img_data = image.ravel()
        
        for i in range(len(binary_data)):
            img_data[i] = int(format(img_data[i], '08b')[:-1] + binary_data[i], 2)
            
        img_data = img_data.reshape(image.shape)
        steg_img = Image.fromarray(np.uint8(img_data))
        steg_img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(steg_img)
        panel2.config(image=img)
        panel2.image = img

root = Tk()
root.title("Steganography App")

frame1 = Frame(root)
frame1.pack(side=LEFT)

frame2 = Frame(root)
frame2.pack(side=RIGHT)

panel1 = Label(frame1, text="No Image")
panel1.pack()

panel2 = Label(frame2, text="No Image")
panel2.pack()

upload_button = Button(frame1, text="Upload Image", command=upload_image)
upload_button.pack()

hide_button = Button(frame2, text="Hide Data", command=hide_data)
hide_button.pack()

data_entry = Entry(root, width=30)
data_entry.pack(side=BOTTOM)
data_entry.insert(0, "Enter text to hide")

root.mainloop()
