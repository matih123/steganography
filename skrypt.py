import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

image_with_hidden_data = "C:\\Users\\matih\\MEGA\\Moje\\Studia\\semestr7\\steganografia\\steganography\\encrypted_image.png"

def string_to_binary(data):
    binary_string = ''.join(format(ord(char), '08b') for char in data)
    return binary_string

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
        image_for_display = resize_image(image, 250)
        photo = ImageTk.PhotoImage(image_for_display)
        panel.configure(image=photo)
        panel.image = photo
        panel.image_name = filename
        panel_encrypted.image = None

def hide_data_in_image_lsb(image, data):
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

def extract_data_from_image_lsb(image):
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

def hide_data_in_image_fft(image, data):
    global original_image
    delimiter = "11111111"
    binary_string = string_to_binary(data) + delimiter
    original_image = image
    image = image.convert('RGB')
    image_array = np.array(image)

    def process_channel(channel):
        freq_domain = np.fft.fft2(channel)
        freq_domain_shifted = np.fft.fftshift(freq_domain)

        height, width = freq_domain_shifted.shape
        constant = 20  # Increased constant value for more distinct encoding

        # Adjust embedding region to a stable part of the frequency domain
        start_x, start_y = width // 4, height // 4
        end_x, end_y = start_x + 50, start_y + 50

        for i, bit in enumerate(binary_string):
            x = start_x + (i % 50)
            y = start_y + (i // 50)
            if y >= end_y or x >= end_x:
                break  # Avoid exceeding the defined range

            if bit == '1':
                freq_domain_shifted[y, x] += constant
            else:
                freq_domain_shifted[y, x] -= constant

        freq_domain_shifted = np.fft.ifftshift(freq_domain_shifted)
        inverse_fft = np.fft.ifft2(freq_domain_shifted)
        result_channel = np.real(inverse_fft)
        return result_channel

    channels_processed = [process_channel(image_array[:, :, i]) for i in range(3)]
    result_image = np.stack(channels_processed, axis=-1)
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    return Image.fromarray(result_image)

def extract_data_from_image_fft(image):
    global original_image
    delimiter = "11111111"
    image = image.convert('RGB')
    image_array = np.array(image)
    original_image_array = np.array(original_image.convert('RGB'))

    extracted_binary_string = ""

    def process_channel(modified_channel, original_channel):
        nonlocal extracted_binary_string
        mod_freq_domain = np.fft.fft2(modified_channel)
        mod_freq_domain_shifted = np.fft.fftshift(mod_freq_domain)

        orig_freq_domain = np.fft.fft2(original_channel)
        orig_freq_domain_shifted = np.fft.fftshift(orig_freq_domain)

        height, width = mod_freq_domain_shifted.shape

        # Define the constant within the function
        constant = 20  # Same constant used for encoding

        # Same embedding region as used in embedding
        start_x, start_y = width // 4, height // 4
        end_x, end_y = start_x + 50, start_y + 50

        for i in range(50 * 50):
            x = start_x + (i % 50)
            y = start_y + (i // 50)
            if y >= end_y or x >= end_x or len(extracted_binary_string) >= len(delimiter) and extracted_binary_string[-len(delimiter):] == delimiter:
                break

            mod_value = mod_freq_domain_shifted[y, x]
            orig_value = orig_freq_domain_shifted[y, x]

            # Use the constant to set the threshold for determining bit value
            threshold = np.real(orig_value) + constant / 2
            bit = '1' if np.real(mod_value) > threshold else '0'
            extracted_binary_string += bit

    for i in range(3):
        process_channel(image_array[:, :, i], original_image_array[:, :, i])
        if len(extracted_binary_string) >= len(delimiter) and extracted_binary_string[-len(delimiter):] == delimiter:
            break

    extracted_binary_string = extracted_binary_string[:-len(delimiter)]
    return binary_string_to_text(extracted_binary_string)

def binary_string_to_text(binary_string):
    text = "".join([chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8)])
    return text

def encrypt_action(encryption_type, event=None):
    if not panel.image_name:
        return
    
    if encryption_type == "LSB":
        hide_data_in_image = hide_data_in_image_lsb
    elif encryption_type == "FFT":
        hide_data_in_image = hide_data_in_image_fft

    data = text_entry.get()
    image = Image.open(panel.image_name)
    new_image = hide_data_in_image(image, data)
    image_for_display = resize_image(new_image, 250)
    photo = ImageTk.PhotoImage(image_for_display)
    panel_encrypted.configure(image=photo)
    panel_encrypted.image = photo
    panel_encrypted.raw_image = new_image
    new_image.save(image_with_hidden_data, format="PNG")
    panel_encrypted.image_name = image_with_hidden_data

def decrypt_action(encryption_type, event=None):
    if not panel.image_name:
        return
    
    if encryption_type == "LSB":
        extract_data_from_image = extract_data_from_image_lsb
    elif encryption_type == "FFT":
        extract_data_from_image = extract_data_from_image_fft

    image = Image.open(panel.image_name)
    try:
        hidden_text = extract_data_from_image(image)
        decrypted_text_entry.delete(0, tk.END) 
        decrypted_text_entry.insert(0, hidden_text) 
    except ValueError as e:
        decrypted_text_entry.delete(0, tk.END) 
        decrypted_text_entry.insert(0, "Could not decrypt the message.")

root = tk.Tk()
root.title("Image Steganography")
root.geometry("1200x600")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)

button_frame = tk.Frame(frame)
button_frame.pack(fill=tk.X, side=tk.TOP)

upload_btn = tk.Button(button_frame, text="Upload Image", command=upload_action)
upload_btn.pack(side=tk.LEFT)

hide_btn = tk.Button(button_frame, text="Hide Text [LSB]", command=lambda: encrypt_action('LSB'))
hide_btn.pack(side=tk.LEFT)

decrypt_btn = tk.Button(button_frame, text="Decrypt Text [LSB]", command=lambda: decrypt_action('LSB'))
decrypt_btn.pack(side=tk.LEFT)

hide_btn1 = tk.Button(button_frame, text="Hide Text [FFT]", command=lambda: encrypt_action('FFT'))
hide_btn1.pack(side=tk.LEFT)

decrypt_btn1 = tk.Button(button_frame, text="Decrypt Text [FFT]", command=lambda: decrypt_action('FFT'))
decrypt_btn1.pack(side=tk.LEFT)

text_var = tk.StringVar()
text_var.set("Enter message to encrypt ...")
text_entry = tk.Entry(frame, width=50, textvariable=text_var)
text_entry.pack(side=tk.TOP, fill=tk.X, expand=True)

# image panel
panel = tk.Label(frame, text="Original Image")
panel.pack(fill=tk.BOTH, expand=True)

decrypted_text_var = tk.StringVar()
decrypted_text_var.set("[Decrypted message will be shown here]")
decrypted_text_entry = tk.Entry(frame, width=50, textvariable=decrypted_text_var)
decrypted_text_entry.pack(side=tk.TOP, fill=tk.X, expand=True)

# Encrypted image panel
panel_encrypted = tk.Label(frame, text="Encrypted Image")
panel_encrypted.pack(fill=tk.BOTH, expand=True)

root.mainloop()
