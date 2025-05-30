import cv2
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

# Setup utama jendela
root = Tk()
root.title("PyImageLab - Pengolahan Citra Digital")
root.geometry("900x650")
root.minsize(700, 550)
root.configure(bg="#87ceeb")

# Variabel global
current_image = None
processed_image = None

# Fungsi untuk menampilkan gambar

def show_image(img):
    w = image_label.winfo_width()
    h = image_label.winfo_height()
    if w <= 1 or h <= 1:
        w, h = 600, 450

    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((w, h), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Fungsi membuka gambar
def open_image():
    global current_image, processed_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        current_image = cv2.imread(file_path)
        processed_image = current_image.copy()
        rgb_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        show_image(rgb_image)
        status_label.config(text="Status: Gambar asli ditampilkan")

        # Aktifkan tombol fitur
        grayscale_btn.config(state=NORMAL)
        binary_btn.config(state=NORMAL)
        save_btn.config(state=NORMAL)
        reset_btn.config(state=NORMAL)
        multiply_btn.config(state=NORMAL)
        nor_btn.config(state=NORMAL)
        histogram_btn.config(state=NORMAL)
        edge_btn.config(state=NORMAL)
        dilate_diag_btn.config(state=NORMAL)
        dilate_hori_btn.config(state=NORMAL)

# Fungsi grayscale
def convert_grayscale():
    global processed_image
    if current_image is not None:
        processed_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        show_image(processed_image)
        status_label.config(text="Status: Gambar dalam mode Grayscale")

# Fungsi biner
def convert_binary():
    global processed_image
    if current_image is not None:
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_image = binary
        show_image(processed_image)
        status_label.config(text="Status: Gambar dalam mode Citra Biner")

# Fungsi reset gambar
def reset_image():
    global processed_image
    if current_image is not None:
        processed_image = current_image.copy()
        rgb_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        show_image(rgb_image)
        status_label.config(text="Status: Gambar asli dikembalikan")

# Fungsi brightness
def multiply_image():
    global processed_image
    if current_image is not None:
        factor = 1.5
        multiplied = cv2.convertScaleAbs(current_image, alpha=factor, beta=0)
        processed_image = multiplied
        rgb_image = cv2.cvtColor(multiplied, cv2.COLOR_BGR2RGB)
        show_image(rgb_image)
        status_label.config(text="Status: Gambar dengan Brightness diperbesar")

# Fungsi operasi logika NOR
def nor_operation():
    global processed_image
    if current_image is not None and processed_image is not None:
        if len(processed_image.shape) == 3:
            proc_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            proc_gray = processed_image

        orig_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        or_result = cv2.bitwise_or(orig_gray, proc_gray)
        nor_result = cv2.bitwise_not(or_result)
        processed_image = nor_result
        show_image(processed_image)
        status_label.config(text="Status: Hasil Operasi Logika NOR ditampilkan")

# Edge Detection
def edge_filter():
    global processed_image
    if current_image is not None:
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed_image = edges
        show_image(edges)
        status_label.config(text="Status: Gambar hasil Edge Detection (Canny)")

# Fungsi dilasi diagonal
def dilate_diagonal():
    global processed_image
    if current_image is not None:
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel_diag = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        dilated = cv2.dilate(binary, kernel_diag, iterations=1)
        processed_image = dilated
        show_image(dilated)
        status_label.config(text="Status: Dilasi dengan kernel diagonal")

# Fungsi dilasi horizontal
def dilate_horizontal():
    global processed_image
    if current_image is not None:
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel_horizontal = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        dilated = cv2.dilate(binary, kernel_horizontal, iterations=1)
        processed_image = dilated
        show_image(dilated)
        status_label.config(text="Status: Dilasi dengan kernel horizontal")

# Simpan gambar
def save_image():
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if file_path:
            try:
                if len(processed_image.shape) == 2:
                    cv2.imwrite(file_path, processed_image)
                else:
                    cv2.imwrite(file_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("Berhasil", "Gambar berhasil disimpan!")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan gambar.\n{e}")

# Tampilkan histogram
def show_histogram():
    if current_image is not None:
        img_rgb = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 4))
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title('Histogram RGB')
        plt.xlabel('Intensitas')
        plt.ylabel('Jumlah Piksel')
        plt.tight_layout()
        plt.show()

# Update saat resize
def on_resize(event):
    global processed_image
    if processed_image is not None:
        if len(processed_image.shape) == 2:
            show_image(processed_image)
        else:
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            show_image(rgb_image)

# === GUI Layout ===
header_frame = Frame(root, bg="#87ceeb")
header_frame.pack(pady=(10,0))

title_label = Label(header_frame, text="URA Image Filter", font=("Segoe UI", 28, "bold"), bg="#87ceeb", fg="#003366")
title_label.pack()

slogan_label = Label(header_frame, text="Solusi Mudah Pengolahan Citra Digital\nCepat, Praktis, dan Informatif", font=("Segoe UI", 12, "italic"), bg="#87ceeb", fg="#003366", justify=CENTER)
slogan_label.pack(pady=(0,10))

button_frame1 = Frame(root, bg="#87ceeb")
button_frame1.pack(pady=3)

button_frame2 = Frame(root, bg="#87ceeb")
button_frame2.pack(pady=3)

Button(button_frame1, text="Buka Gambar", command=open_image, bg="#1976d2", fg="white", padx=12, pady=6).grid(row=0, column=0, padx=8)
grayscale_btn = Button(button_frame1, text="Grayscale", command=convert_grayscale, state=DISABLED, padx=12, pady=6)
grayscale_btn.grid(row=0, column=1, padx=8)
binary_btn = Button(button_frame1, text="Citra Biner", command=convert_binary, state=DISABLED, padx=12, pady=6)
binary_btn.grid(row=0, column=2, padx=8)
multiply_btn = Button(button_frame1, text="Brightness", command=multiply_image, state=DISABLED, padx=12, pady=6)
multiply_btn.grid(row=0, column=3, padx=8)

nor_btn = Button(button_frame2, text="Operasi NOR", command=nor_operation, state=DISABLED, padx=12, pady=6)
nor_btn.grid(row=0, column=0, padx=8)
edge_btn = Button(button_frame2, text="Edge Filter", command=edge_filter, state=DISABLED, padx=12, pady=6)
edge_btn.grid(row=0, column=1, padx=8)
reset_btn = Button(button_frame2, text="Reset Gambar", command=reset_image, state=DISABLED, padx=12, pady=6)
reset_btn.grid(row=0, column=2, padx=8)
save_btn = Button(button_frame2, text="Simpan Gambar", command=save_image, state=DISABLED, padx=12, pady=6)
save_btn.grid(row=0, column=3, padx=8)
histogram_btn = Button(button_frame2, text="Histogram", command=show_histogram, state=DISABLED, padx=12, pady=6)
histogram_btn.grid(row=0, column=4, padx=8)

dilate_diag_btn = Button(button_frame2, text="Dilasi Diagonal", command=dilate_diagonal, state=DISABLED, padx=12, pady=6)
dilate_diag_btn.grid(row=0, column=5, padx=8)

dilate_hori_btn = Button(button_frame2, text="Dilasi Horizontal", command=dilate_horizontal, state=DISABLED, padx=12, pady=6)
dilate_hori_btn.grid(row=0, column=6, padx=8)

image_frame = Frame(root, bg="#87ceeb")
image_frame.pack(fill=BOTH, expand=True, padx=20, pady=5)

image_label = Label(image_frame, bg="white", relief=SUNKEN, bd=2)
image_label.pack(fill=BOTH, expand=True)

status_label = Label(image_frame, text="Status: Menunggu gambar dibuka", font=("Segoe UI", 10), bg="white", fg="black")
status_label.place(x=5, y=5)

root.bind("<Configure>", on_resize)

root.mainloop()
