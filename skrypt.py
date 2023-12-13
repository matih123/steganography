import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import itertools
import cv2
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import base64

image_with_hidden_data = "C:\\Users\\matih\\MEGA\\Moje\\Studia\\semestr7\\steganografia\\steganography\\encrypted-image.png"

def reverse_shift_characters(s, shift):
    """Reverses the shift of each character in the string by 'shift' number of places."""
    reversed_string = ""
    for char in s:
        # Reverse shift only if it's an alphabet character
        if char.isalpha():
            # Determine if it's uppercase or lowercase for correct shifting
            start = ord('A') if char.isupper() else ord('a')
            # Reverse shift the character and wrap around the alphabet if necessary
            reversed_char = chr((ord(char) - start - shift) % 26 + start)
            reversed_string += reversed_char
        else:
            # Keep non-alphabet characters as they are
            reversed_string += char
    return reversed_string

# Klucz AES = "STEGANOGRAFIA_STATYCZNY_KLUCZ_BEZPIECZENSTWA"
obfuscated_string = "VWHJDQRJUDILD_VWDWBFCQB_NOXFC_EHCSLHFCHQVWZD"
aes_key = reverse_shift_characters(obfuscated_string, 3) # get non obfuscated AES key on runtime

def aes_encrypt(data, key):
    iv = os.urandom(16) 
    key = key.ljust(32)[:32] 

    cipher = Cipher(algorithms.AES(key.encode()), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    padded_data = data + ' ' * (16 - len(data) % 16)
    
    encrypted_data = encryptor.update(padded_data.encode()) + encryptor.finalize()
    encrypted_data_with_iv = iv + encrypted_data
    return base64.b64encode(encrypted_data_with_iv).decode()

def aes_decrypt(encrypted_base64_data, key):
    key = key.ljust(32)[:32]
    encrypted_data_with_iv = base64.b64decode(encrypted_base64_data)

    iv = encrypted_data_with_iv[:16]
    encrypted_data = encrypted_data_with_iv[16:]

    cipher = Cipher(algorithms.AES(key.encode()), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    return decrypted_data.rstrip().decode()

class DCT():
    quant = np.array([[16,11,10,16,24,40,51,61],
                        [12,12,14,19,26,58,60,55],  
                        [14,13,16,24,40,57,69,56],
                        [14,17,22,29,51,87,80,62],
                        [18,22,37,56,68,109,103,77],
                        [24,35,55,64,81,104,113,92],
                        [49,64,78,87,103,121,120,101],
                        [72,92,95,98,112,100,103,99]])

    def __init__(self):
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0

    def encode_image(self,img,secret_msg):

        secret=secret_msg
        self.message = str(len(secret))+'*'+secret
        self.bitMess = self.toBits()

        row,col = img.shape[:2]

        self.oriRow, self.oriCol = row, col  
        if((col/8)*(row/8)<len(secret)):
            print("Error: Message too large to encode in image")
            return False

        if row%8 != 0 or col%8 != 0:
            img = self.addPadd(img, row, col)
        
        row,col = img.shape[:2]
        channels = cv2.split(img)
        if len(channels) == 3:
            bImg,gImg,rImg = channels
        elif len(channels) == 4:
            bImg,gImg,rImg,_ = channels


        bImg = np.float32(bImg)

        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j,i) in itertools.product(range(0,row,8), range(0,col,8))]

        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]

        quantizedDCT = [np.round(dct_Block/self.quant) for dct_Block in dctBlocks]

        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC= DC-255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex+1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        sImgBlocks = [quantizedBlock *self.quant+128 for quantizedBlock in quantizedDCT]

        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)

        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        return sImg

    def decode_image(self,img):
        row,col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0

        bImg,gImg,rImg = cv2.split(img)

        bImg = np.float32(bImg)

        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]    

        quantizedDCT = [img_Block/self.quant for img_Block in imgBlocks]
        i=0

        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 1:
                buff+=(0 & 1) << (7-i)
            elif DC[7] == 0:
                buff+=(1&1) << (7-i)
            i=1+i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i =0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize))+1:]
        sImgBlocks = [quantizedBlock *self.quant+128 for quantizedBlock in quantizedDCT]
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        return ''
      

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def addPadd(self, img, row, col):
        return cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)) )
    
    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8,'0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8,'0')
        return bits
    
def encrypt_action_dct(event=None):
    if not panel.image_name:
        return
    
    data = text_entry.get()
    data_hash = hashlib.sha256(data.encode()).hexdigest()
    dct_img = cv2.imread(panel.image_name, cv2.IMREAD_UNCHANGED)
    dct_img_encoded = DCT().encode_image(dct_img, aes_encrypt(f"{data}{data_hash}", aes_key))
    
    image_for_display = Image.fromarray(cv2.cvtColor(dct_img_encoded, cv2.COLOR_BGR2RGB))
    image_for_display = resize_image(image_for_display, 500)
    
    photo = ImageTk.PhotoImage(image_for_display)
    panel_encrypted.configure(image=photo)
    panel_encrypted.image = photo
    
    cv2.imwrite(image_with_hidden_data, dct_img_encoded)
    panel_encrypted.image_name = image_with_hidden_data

def decrypt_action_dct(event=None):
    if not panel.image_name:
        return

    dct_img = cv2.imread(panel.image_name, cv2.IMREAD_UNCHANGED)
    dct_hidden_text = aes_decrypt(DCT().decode_image(dct_img), aes_key)

    hidden_data = dct_hidden_text[:-64]
    hidden_data_hash1 = dct_hidden_text[-64:]
    hidden_data_hash2 = hashlib.sha256(hidden_data.encode()).hexdigest()
    if hidden_data_hash1 == hidden_data_hash2:
        hidden_data += " [integrity validated]"
    else:
        hidden_data += " [ERROR: integrity validation]"

    decrypted_text_entry.delete(0, tk.END)
    decrypted_text_entry.insert(0, hidden_data)

def resize_image(image, max_size):
    r1 = image.width / max_size
    r2 = image.height / max_size
    ratio = max(r1, r2)
    new_size = (int(image.width / ratio), int(image.height / ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def upload_action(event=None):
    filename = filedialog.askopenfilename()
    if filename:
        image = Image.open(filename)
        image_for_display = resize_image(image, 500)
        photo = ImageTk.PhotoImage(image_for_display)
        panel.configure(image=photo)
        panel.image = photo
        panel.image_name = filename
        panel_encrypted.image = None

def hide_data_in_image(image, data):

    data += "#####"
    binary_data = ''.join([format(ord(i), '08b') for i in data])
    data_len = len(binary_data)

    imageData = np.array(image)
    flat = imageData.flatten()
    idx = 0

    for i in range(data_len):
        flat[idx] = (flat[idx] & ~1) | int(binary_data[i])
        idx += 1

    imageData = flat.reshape(imageData.shape)
    new_image = Image.fromarray(imageData)
    return new_image

def extract_data_from_image(image):
    imageData = np.array(image)
    flat = imageData.flatten()

    decoded_data = ""
    bits = ""

    for i in flat:
        bits += str(i & 1)
        if len(bits) == 8:
            byte = chr(int(bits, 2))
            decoded_data += byte
            bits = ""
            if decoded_data[-5:] == "#####": 
                return decoded_data[:-5]
    return decoded_data

def encrypt_action(event=None):
    if not panel.image_name:
        return
    
    data = text_entry.get()
    data_hash = hashlib.sha256(data.encode()).hexdigest()

    image = Image.open(panel.image_name)
    new_image = hide_data_in_image(image, aes_encrypt(f"{data}{data_hash}", aes_key))
    image_for_display = resize_image(new_image, 500)
    photo = ImageTk.PhotoImage(image_for_display)
    panel_encrypted.configure(image=photo)
    panel_encrypted.image = photo
    panel_encrypted.raw_image = new_image
    # Save the actual encrypted image at full size in a lossless format like PNG
    new_image.save(image_with_hidden_data, format="PNG")
    #save in path filename
    panel_encrypted.image_name = image_with_hidden_data

def decrypt_action(event=None):
    if not panel.image_name:
        return
    image = Image.open(panel.image_name)
    try:
        hidden = aes_decrypt(extract_data_from_image(image), aes_key)
        hidden_data = hidden[:-64]
        hidden_data_hash1 = hidden[-64:]
        hidden_data_hash2 = hashlib.sha256(hidden_data.encode()).hexdigest()
        if hidden_data_hash1 == hidden_data_hash2:
            hidden_data += " [integrity validated]"
        else:
            hidden_data += " [ERROR: integrity validation]"

        decrypted_text_entry.delete(0, tk.END) 
        decrypted_text_entry.insert(0, hidden_data) 
    except ValueError as e:
        decrypted_text_entry.delete(0, tk.END) 
        decrypted_text_entry.insert(0, "Could not decrypt the message.")

root = tk.Tk()
root.title("Image Steganography")
root.geometry("1300x700")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)

button_frame = tk.Frame(frame)
button_frame.pack(fill=tk.X, side=tk.TOP)

upload_btn = tk.Button(button_frame, text="Upload Image", font=("Helvetica", 12), command=upload_action)
upload_btn.pack(side=tk.LEFT)

hide_btn = tk.Button(button_frame, text="Hide Text (LSB)", font=("Helvetica", 12), command=encrypt_action)
hide_btn.pack(side=tk.LEFT)

decrypt_btn = tk.Button(button_frame, text="Decrypt Text (LSB)", font=("Helvetica", 12), command=decrypt_action)
decrypt_btn.pack(side=tk.LEFT)

hide_btn_dct = tk.Button(button_frame, text="Hide Text (DCT)", font=("Helvetica", 12), command=encrypt_action_dct)
hide_btn_dct.pack(side=tk.LEFT)

decrypt_btn_dct = tk.Button(button_frame, text="Decrypt Text (DCT)", font=("Helvetica", 12), command=decrypt_action_dct)
decrypt_btn_dct.pack(side=tk.LEFT)

text_var = tk.StringVar()
text_var.set("Enter message to encrypt ...")
text_entry = tk.Entry(frame, width=50, font=("Helvetica", 14), textvariable=text_var)
text_entry.pack(side=tk.TOP, fill=tk.X, expand=True)

# image panel
panel = tk.Label(frame, text="Original Image", font=("Helvetica", 14))
panel.pack(fill=tk.BOTH, expand=True)

decrypted_text_var = tk.StringVar()
decrypted_text_var.set("[Decrypted message will be shown here]")
decrypted_text_entry = tk.Entry(frame, width=50, font=("Helvetica", 14), textvariable=decrypted_text_var)
decrypted_text_entry.pack(side=tk.TOP, fill=tk.X, expand=True)

# Encrypted image panel
panel_encrypted = tk.Label(frame, text="Encrypted Image", font=("Helvetica", 14))
panel_encrypted.pack(fill=tk.BOTH, expand=True)

root.mainloop()
