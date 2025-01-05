import sys
import speech_recognition as sr
import torch
import transformers
import pyttsx3
import threading
import queue
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, QWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class SpeechToSpeechBot(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Language Model Setup
        self.model = transformers.T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('google/flan-t5-large')

        # Text-to-Speech Setup
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 200)

        # Webcam Setup
        self.video_capture = cv2.VideoCapture(0)
        self.is_capturing = False

        # Setup UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Speech-to-Speech Bot')
        self.setGeometry(100, 100, 800, 600)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Video Label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Control Button
        self.capture_button = QPushButton('Start Capture', self)
        self.capture_button.clicked.connect(self.toggle_capture)
        main_layout.addWidget(self.capture_button)

        # Log Area
        self.log_area = QTextEdit(self)
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_area)

        # Video Update Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(25)  # 40 FPS

    def toggle_capture(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.capture_button.setText('Stop Capture')
            threading.Thread(target=self.process_interaction, daemon=True).start()
        else:
            self.is_capturing = False
            self.capture_button.setText('Start Capture')

    def update_video_feed(self):
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def log_message(self, message):
        self.log_area.append(message)

    def capture_speech(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.log_message("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                return text.strip()
            except sr.UnknownValueError:
                self.log_message("Could not understand audio")
                return None
            except sr.RequestError:
                self.log_message("Speech recognition service error")
                return None

    def generate_response(self, input_text):
        # [Previous implementation remains the same]
        if not input_text:
            return "I'm sorry, but I couldn't hear what you said."

        enhanced_prompt = f"Provide a comprehensive explanation about: {input_text}"

        try:
            input_ids = self.tokenizer(enhanced_prompt, return_tensors="pt", max_length=512, truncation=True).input_ids

            outputs = self.model.generate(
                input_ids, 
                max_length=300,
                min_length=100,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            self.log_message(f"Response generation error: {e}")
            return "I encountered an error generating a response."

    def speak_response(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            self.log_message(f"Speaking: {text}")
        except Exception as e:
            self.log_message(f"Speech synthesis error: {e}")

    def process_interaction(self):
        while self.is_capturing:
            speech_text = self.capture_speech()
            
            if speech_text:
                self.log_message(f"Detected speech: {speech_text}")
                response_text = self.generate_response(speech_text)
                self.log_message(f"Generated response: {response_text}")
                self.speak_response(response_text)
            
            time.sleep(1)

    def closeEvent(self, event):
        self.video_capture.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    bot = SpeechToSpeechBot()
    bot.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()