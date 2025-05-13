import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Toplevel, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("leaf_disease_model.h5")
IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image():
    global file_path
    if file_path:
        update_status("Classifying...")
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]
        if prediction >= 0.5:
            result = "Healthy"
            color = "#388e3c"
        else:
            result = "Unhealthy"
            color = "#d32f2f"
        result_label.config(text=f"Prediction: {result} ({prediction:.2f})", fg=color)
        confidence_bar['value'] = int(prediction * 100)
        update_status("Classification complete.")

def upload_image():
    global file_path, img_original
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img_original = Image.open(file_path)
        img_resized = img_original.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_resized)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="", fg="#333")
        confidence_bar['value'] = 0
        update_status("Image uploaded.")

def clear_all():
    global file_path
    file_path = None
    image_label.config(image="")
    image_label.image = None
    result_label.config(text="")
    confidence_bar['value'] = 0
    update_status("Cleared.")

def preview_image():
    if file_path:
        top = Toplevel(root)
        top.title("Full Image Preview")
        img_zoom = img_original.resize((500, 500))
        img_tk = ImageTk.PhotoImage(img_zoom)
        label = Label(top, image=img_tk)
        label.image = img_tk
        label.pack()

def update_status(msg):
    status_label.config(text=msg)

# GUI Setup
root = tk.Tk()
root.title("Leaf Disease Detection")
root.geometry("550x750")
root.configure(bg="#e9eff6")

file_path = None
img_original = None

# Header
header = Label(root, text="🌿 Leaf Disease Detector", font=("Helvetica", 24, "bold"), bg="#e9eff6", fg="#2c3e50")
header.pack(pady=20)

# Main container
container = Frame(root, bg="white", bd=2, relief="groove")
container.pack(padx=20, pady=10, fill="both", expand=True)

# Image display
image_label = Label(container, bg="white", cursor="hand2")
image_label.pack(pady=15)
image_label.bind("<Button-1>", lambda e: preview_image())

# Upload button
upload_btn = Button(container, text="Upload Leaf Image", command=upload_image,
                    font=("Helvetica", 13), bg="#4caf50", fg="white", padx=15, pady=8, bd=0, relief="ridge", cursor="hand2")
upload_btn.pack(pady=10)

# Predict button
predict_btn = Button(container, text="Analyze Healthy or Diseased", command=classify_image,
                     font=("Helvetica", 13), bg="#2196f3", fg="white", padx=15, pady=8, bd=0, relief="ridge", cursor="hand2")
predict_btn.pack(pady=5)

# Clear button
clear_btn = Button(container, text="Clear", command=clear_all,
                   font=("Helvetica", 12), bg="#f44336", fg="white", padx=10, pady=5, bd=0, relief="ridge", cursor="hand2")
clear_btn.pack(pady=5)

# Prediction result
result_label = Label(container, text="", font=("Helvetica", 16, "bold"), bg="white")
result_label.pack(pady=10)

# Confidence progress bar
Label(container, text="Confidence", font=("Helvetica", 11), bg="white").pack()
confidence_bar = ttk.Progressbar(container, orient="horizontal", length=250, mode="determinate")
confidence_bar.pack(pady=5)

# Footer
footer = Label(root, text="Developed by Ziad Faris, Bahaa, Ahmad, Ahmad Osama • 2025", font=("Helvetica", 10), bg="#e9eff6", fg="#555")
footer.pack(pady=5)

# Status bar
status_label = Label(root, text="Waiting for input...", font=("Helvetica", 9), bg="#e9eff6", fg="#777")
status_label.pack(pady=5)

root.mainloop()
