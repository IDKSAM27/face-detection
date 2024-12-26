# remember to add pyqt5 in requirements.txt

import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import face_recognition

# Worker Thread for Face Detection
class DetectionThread(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self, video_source, face_encodings, face_names, parent=None):
        super().__init__(parent)
        self.video_source = video_source
        self.face_encodings = face_encodings
        self.face_names = face_names
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_source)

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings_frame = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_frame):
                matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.face_names[first_match_index]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Label the face
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Convert the frame to QImage for display
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.frame_signal.emit(qt_image)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# Main Application Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection and Recognition")
        self.setGeometry(100, 100, 800, 660)

        self.face_encodings = []
        self.face_names = []
        self.detection_thread = None

        # UI Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel("Video Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(784, 480)
        self.layout.addWidget(self.video_label)

        self.load_button = QPushButton("Load Face Dataset")
        self.load_button.clicked.connect(self.load_face_dataset)
        self.layout.addWidget(self.load_button)

        self.open_video_button = QPushButton("Open Video")
        self.open_video_button.clicked.connect(self.open_video)
        self.layout.addWidget(self.open_video_button)

        self.start_webcam_button = QPushButton("Start Webcam")
        self.start_webcam_button.clicked.connect(self.start_webcam)
        self.layout.addWidget(self.start_webcam_button)

        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.layout.addWidget(self.stop_button)

    def load_face_dataset(self):
        dataset_path = QFileDialog.getExistingDirectory(self, "Select Face Dataset Folder")
        if not dataset_path:
            QMessageBox.information(self, "Info", "No folder selected.")
            return

        self.face_encodings.clear()
        self.face_names.clear()

        try:
            for file_name in os.listdir(dataset_path):
                if file_name.endswith(('.jpg', '.png')):
                    image_path = os.path.join(dataset_path, file_name)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)

                    if encoding:
                        self.face_encodings.append(encoding[0])
                        self.face_names.append(os.path.splitext(file_name)[0])

            QMessageBox.information(self, "Info", f"Loaded {len(self.face_names)} faces from the dataset.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def open_video(self):
        video_path = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")[0]
        if not video_path:
            QMessageBox.information(self, "Info", "No video file selected.")
            return

        self.start_detection(video_path)

    def start_webcam(self):
        self.start_detection(0)

    def start_detection(self, source):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()

        self.detection_thread = DetectionThread(source, self.face_encodings, self.face_names)
        self.detection_thread.frame_signal.connect(self.update_frame)
        self.detection_thread.start()

    def stop_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
        self.video_label.clear()

    def update_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

# Main Application Entry Point
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


# Add a database to the detection


"""
# For comparison, below system uses tkinter whereas above uses PyQt
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import face_recognition


# Initialize the main application window
root = tk.Tk()
root.title("Face Detection and Recognition")
root.geometry("800x660")
root.resizable(width=True, height=True)

# Global variables
cap = None
stop_thread = False
face_encodings = []
face_names = []

# Load pre-trained face dataset
def load_face_dataset():
    global face_encodings, face_names
    dataset_path = filedialog.askdirectory(title="Select Face Dataset Folder")

    if not dataset_path:
        messagebox.showinfo("Info", "No folder selected.")
        return

    face_encodings.clear()
    face_names.clear()

    try:
        for file_name in os.listdir(dataset_path):
            if file_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(dataset_path, file_name)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)

                if encoding:
                    face_encodings.append(encoding[0])
                    face_names.append(os.path.splitext(file_name)[0])

        messagebox.showinfo("Info", f"Loaded {len(face_names)} faces from the dataset.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")

# Open a video file

def open_video():
    global cap, stop_thread
    stop_thread = False
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", ".mp4"), ("All files", ".*")))

    if video_path:
        print(f"Selected video path: {video_path}") # Debugging output
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            print("Error: Could not open video.") # Debugging output  
        else:
            print("Video opened successfully.") # Debugging output
            threading.Thread(target=detect_faces).start()
    else:
        messagebox.showinfo("Info", "No video file selected.")

# Start the webcam
def start_webcam():
    global cap, stop_thread
    stop_thread = True
    if cap is not None:
        cap.release()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        print(f"Error: Could not open webcam.") # Debugging output
        return

    print("Webcam opened successfully.") # Debugging output
    stop_thread = False
    threading.Thread(target=detect_faces).start()


# Detect and recognize faces
def detect_faces():
    global cap, stop_thread

    while cap.isOpened() and not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings_frame = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_frame):
            matches = face_recognition.compare_faces(face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = face_names[first_match_index]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Label the face
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Resize the frame to fit the label
        resized_frame = cv2.resize(frame, (lbl_video.winfo_width(), lbl_video.winfo_height()))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        root.update()

    cap.release()
    cv2.destroyAllWindows()



# Stop the detection thread
def stop_detection():
    global stop_thread
    stop_thread = True
    if cap is not None:
        cap.release()
    lbl_video.configure(image='')
    print("Detection stopped.")


# GUI Elements
lbl_video = tk.Label(root, width=30, height=30) # Set desired width or height of the video loaded
lbl_video.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

btn_load_dataset = tk.Button(button_frame, text="Load Face Dataset", command=load_face_dataset)
btn_load_dataset.pack(side=tk.LEFT, padx=10)

btn_open_video = tk.Button(button_frame, text="Open Video", command=open_video)
btn_open_video.pack(side=tk.LEFT, padx=10)

btn_start_webcam = tk.Button(button_frame, text="Start Webcam", command=start_webcam)
btn_start_webcam.pack(side=tk.LEFT, padx=10)

btn_stop_detection = tk.Button(button_frame, text="Stop Detection", command=stop_detection)
btn_stop_detection.pack(side=tk.LEFT, padx=10)

root.mainloop()
# Update the dimensions of the video footage upon loading
"""