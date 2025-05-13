import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Toplevel, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load model
model = tf.keras.models.load_model("leaf_disease_model.h5")
IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def classify_image():
    global file_path
    if file_path:
        update_status("Classifying...")
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]
        result = "Healthy" if prediction >= 0.5 else "Unhealthy"
        color = "#00e676" if prediction >= 0.5 else "#ff5252"
        result_label.config(text=f"Prediction: {result} ({prediction:.2f})", fg=color)
        confidence_bar['value'] = int(prediction * 100)
        update_status("Classification complete.")

def upload_image():
    global file_path, img_original
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img_original = Image.open(file_path)
        img_resized = img_original.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img_resized)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="", fg="white")
        confidence_bar['value'] = 0
        edge_label.config(image="")
        update_status("Image uploaded.")

def clear_all():
    global file_path
    file_path = None
    image_label.config(image="")
    image_label.image = None
    edge_label.config(image="")
    result_label.config(text="")
    confidence_bar['value'] = 0
    update_status("Cleared.")

def preview_image():
    if file_path:
        top = Toplevel(root)
        top.title("Full Image Preview")
        img_zoom = img_original.resize((700, 700))
        img_tk = ImageTk.PhotoImage(img_zoom)
        label = Label(top, image=img_tk)
        label.image = img_tk
        label.pack()

def show_edge_detection():
    if file_path:
        update_status("Applying edge detection...")
        img_cv = cv2.imread(file_path)
        img_cv = cv2.resize(img_cv, (400, 400))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(edges_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        edge_label.config(image=img_tk)
        edge_label.image = img_tk
        update_status("Edge detection applied.")

def update_status(msg):
    status_label.config(text=msg)

# --- Main Window ---
root = tk.Tk()
root.title("🌿 Leaf Disease Detection")
root.configure(bg="#121212")

# Centered 1200x900 window
w, h = 500, 900
screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
x = (screen_w // 2) - (w // 2)
y = (screen_h // 2) - (h // 2)
root.geometry(f"{w}x{h}+{x}+{y}")

file_path = None
img_original = None

# Main Content Frame
content_frame = Frame(root, bg="#121212")
content_frame.pack(expand=True)

# Header
header = Label(content_frame, text="🌿 Leaf Disease Detector", font=("Helvetica", 30, "bold"), bg="#121212", fg="white")
header.pack(pady=10)

# Image Display Frame
image_frame = Frame(content_frame, bg="#121212")
image_frame.pack(pady=20)

image_label = Label(image_frame, bg="#121212", cursor="hand2")
image_label.pack(side="left", padx=20)
image_label.bind("<Button-1>", lambda e: preview_image())

edge_label = Label(image_frame, bg="#121212")
edge_label.pack(side="left", padx=20)

# Confidence Bar
Label(content_frame, text="Confidence", font=("Helvetica", 14), bg="#121212", fg="#cccccc").pack()
style = ttk.Style()
style.theme_use('default')
style.configure("TProgressbar", thickness=20, troughcolor="#333333", background="#00c853")
confidence_bar = ttk.Progressbar(content_frame, orient="horizontal", length=400, mode="determinate", style="TProgressbar")
confidence_bar.pack(pady=10)

# Result label
result_label = Label(content_frame, text="", font=("Helvetica", 18, "bold"), bg="#121212", fg="white")
result_label.pack(pady=10)

# Button styling
def styled_button(text, command, color):
    return Button(content_frame, text=text, command=command,
                  font=("Helvetica", 13), bg=color, fg="white",
                  padx=15, pady=6, bd=0, relief="ridge", cursor="hand2")

# Buttons
styled_button("Upload Leaf Image", upload_image, "#3949ab").pack(pady=6)
styled_button("Analyze Healthy or Diseased", classify_image, "#1e88e5").pack(pady=6)
styled_button("Show Edge Detection", show_edge_detection, "#00acc1").pack(pady=6)
styled_button("Clear", clear_all, "#757575").pack(pady=15)

# Footer
footer = Label(content_frame, text="Developed by Ziad Faris, Bahaa, Ahmad, Ahmad Osama • 2025",
               font=("Helvetica", 11), bg="#121212", fg="#666666")
footer.pack(pady=20)

# Status Bar
status_label = Label(root, text="Waiting for input...", font=("Helvetica", 10), bg="#121212", fg="#888888")
status_label.pack(pady=5)

root.mainloop()
