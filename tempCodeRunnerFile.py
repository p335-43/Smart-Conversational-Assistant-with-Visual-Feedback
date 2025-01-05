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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

class SignalEmitter(QObject):
    log_signal = pyqtSignal(str)

class SpeechToSpeechBot(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Signal Emitter for thread-safe logging
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.log_signal.connect(self.log_message)
        
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
        self.stop_event = threading.Event()
        
        # State flags
        self.is_speaking = False
        self.is_listening = False

        # Setup UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Advanced Speech-to-Speech Bot')
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
            self.stop_event.clear()
            self.capture_button.setText('Stop Capture')
            # Use a new thread for continuous interaction
            threading.Thread(target=self.robust_continuous_interaction, daemon=True).start()
        else:
            self.is_capturing = False
            self.stop_event.set()
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
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                self.signal_emitter.log_signal.emit("Listening...")
                self.is_listening = True
                
                try:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    
                    text = self.recognizer.recognize_google(audio)
                    return text.strip()
                except sr.UnknownValueError:
                    self.signal_emitter.log_signal.emit("Could not understand audio")
                    return None
                except sr.RequestError as e:
                    self.signal_emitter.log_signal.emit(f"Speech recognition service error: {e}")
                    return None
                finally:
                    self.is_listening = False
        except Exception as e:
            self.signal_emitter.log_signal.emit(f"Speech capture error: {e}")
            self.is_listening = False
            return None

    def generate_response(self, input_text):
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
            self.signal_emitter.log_signal.emit(f"Response generation error: {e}")
            return "I encountered an error generating a response."

    def speak_response(self, text):
        try:
            self.is_speaking = True
            self.signal_emitter.log_signal.emit(f"Speaking: {text}")
            
            # Use a loop to allow stopping
            def on_done():
                self.is_speaking = False
            
            self.engine.connect('finished-utterance', on_done)
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.signal_emitter.log_signal.emit(f"Speech synthesis error: {e}")
            self.is_speaking = False

    def robust_continuous_interaction(self):
        while not self.stop_event.is_set():
            try:
                # Wait if currently speaking
                while self.is_speaking and not self.stop_event.is_set():
                    time.sleep(0.1)
                
                # Break if stop event is set
                if self.stop_event.is_set():
                    break
                
                # Capture speech
                speech_text = self.capture_speech()
                
                # If speech detected, process it
                if speech_text:
                    self.signal_emitter.log_signal.emit(f"Detected speech: {speech_text}")
                    
                    # Generate response
                    response_text = self.generate_response(speech_text)
                    self.signal_emitter.log_signal.emit(f"Generated response: {response_text}")
                    
                    # Speak response
                    self.speak_response(response_text)
                
                # Small sleep to prevent tight loop
                time.sleep(0.1)
            
            except Exception as e:
                self.signal_emitter.log_signal.emit(f"Interaction error: {e}")
                # Wait a bit before retrying to prevent rapid error loops
                time.sleep(1)

    def closeEvent(self, event):
        self.stop_event.set()
        self.video_capture.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    bot = SpeechToSpeechBot()
    bot.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()