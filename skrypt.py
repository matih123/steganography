from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

def save_image():
    if hasattr(panel2, 'steg_img'):
        save_path = filedialog.asksaveasfilename(defaultextension=".png")
        if save_path:
            panel2.steg_img.save(save_path)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((250, 250))
        img = ImageTk.PhotoImage(image)
        panel1.config(image=img)
        panel1.image = img
        panel1.file_path = file_path

def hide_data_lsb():
    data = data_entry.get()
    if hasattr(panel1, 'file_path'):
        # LSB algorithm
        binary_data = ''.join(format(ord(i), '08b') for i in data)
        
        image = cv2.imread(panel1.file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_data = image.ravel()

        for i in range(len(binary_data)):
            img_data[i] = int(format(img_data[i], '08b')[:-1] + binary_data[i], 2)

        img_data = img_data.reshape(image.shape)
        steg_img = Image.fromarray(np.uint8(img_data))
        panel2.steg_img = steg_img
        steg_img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(steg_img)
        panel2.config(image=img)
        panel2.image = img

def reveal_data_lsb():
    if hasattr(panel1, 'image'):
        image = np.array(panel1.image)
        img_data = image.ravel()
        binary_data = ''
        for i in img_data:
            print(i)
            binary_data += format(i, '08b')[-1]
        
        decrypted_data = ''
        for i in range(0, len(binary_data), 8):
            byte = binary_data[i:i+8]
            decrypted_data += chr(int(byte, 2))
        
        if decrypted_data.find('\x00') != -1:
            decrypted_data = decrypted_data[:decrypted_data.find('\x00')]
        
        data_entry.delete(0, END)
        data_entry.insert(0, decrypted_data)

def prediction_error_expansion(image_path, secret_text):
    # Wczytanie obrazu i konwersja na skalę szarości
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Inicjalizacja zmiennych
    secret_bin = ''.join(format(ord(char), '08b') for char in secret_text)
    idx = 0
    data_len = len(secret_bin)
    
    # Iteracja przez piksele obrazu
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if idx < data_len:
                # Obliczenie błędu predykcji
                prediction = int((img_array[i-1, j] + img_array[i, j-1]) / 2) if i > 0 and j > 0 else img_array[i, j]
                error = img_array[i, j] - prediction
                
                # Ekspansja błędu predykcji
                if secret_bin[idx] == '1':
                    error = error | 1
                else:
                    error = error & ~1
                
                # Aktualizacja wartości piksela
                img_array[i, j] = prediction + error
                
                idx += 1
    
    # Zapisanie nowego obrazu
    new_img = Image.fromarray(img_array.astype('uint8'), 'L')
    new_img.save("stego_image.png")

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

reveal_button = Button(frame2, text="Reveal Data [LSB]", command=reveal_data_lsb)
reveal_button.pack()

save_button = Button(frame2, text="Save Image", command=save_image)
save_button.pack()

upload_button = Button(frame1, text="Upload Image", command=upload_image)
upload_button.pack()

hide_button = Button(frame2, text="Hide Data [LSB]", command=hide_data_lsb)
hide_button.pack()

data_entry = Entry(root, width=30)
data_entry.pack(side=BOTTOM)
data_entry.insert(0, "Enter text to hide")

root.mainloop()
