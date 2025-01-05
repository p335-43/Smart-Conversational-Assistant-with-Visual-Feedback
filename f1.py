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
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class SignalEmitter(QObject):
    log_signal = pyqtSignal(str)

class SpeechToSpeechBot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.log_signal.connect(self.log_message)
        
        # Initialize speech recognition with shorter timeout
        self.recognizer = sr.Recognizer()
        self.recognizer.operation_timeout = 1.0  # Set shorter timeout for recognition
        self.microphone = sr.Microphone()
        
        # Use smaller, faster model
        self.model = transformers.T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('google/flan-t5-small')
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimize model for inference
        self.model.eval()
        torch.set_grad_enabled(False)
        
        # Initialize text-to-speech with faster rate
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 250)  # Increased speech rate
        
        self.video_capture = cv2.VideoCapture(0)
        self.is_capturing = False
        self.stop_event = threading.Event()
        self.is_speaking = False
        self.is_listening = False
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.initUI()

    def log_message(self, message):
        """Add message to the log area."""
        self.log_area.append(message)

    def initUI(self):
        # UI initialization remains the same
        self.setWindowTitle('Fast Speech-to-Speech Bot')
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.capture_button = QPushButton('Start Capture', self)
        self.capture_button.clicked.connect(self.toggle_capture)
        main_layout.addWidget(self.capture_button)

        self.log_area = QTextEdit(self)
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_area)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(25)

    def toggle_capture(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.stop_event.clear()
            self.capture_button.setText('Stop Capture')
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

    def capture_speech(self):
        try:
            with self.microphone as source:
                # Reduced ambient noise adjustment time
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                
                self.signal_emitter.log_signal.emit("Listening...")
                self.is_listening = True
                
                try:
                    # Reduced phrase time limit
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=2.0)
                    
                    # Use faster recognition service
                    text = self.recognizer.recognize_google(audio)
                    return text.strip()
                except sr.UnknownValueError:
                    return None
                except sr.RequestError:
                    return None
                finally:
                    self.is_listening = False
        except Exception as e:
            self.is_listening = False
            return None

    def generate_response(self, input_text):
        if not input_text:
            return "I didn't catch that."

        try:
            # Simplified prompt
            input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=128, 
                                     truncation=True).input_ids.to(self.device)

            # Optimized generation parameters for speed
            outputs = self.model.generate(
                input_ids,
                max_length=50,  # Shorter response
                min_length=10,
                num_return_sequences=1,
                do_sample=True,
                top_k=20,
                top_p=0.9,
                temperature=0.7,
                early_stopping=True
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            return "Processing error occurred."

    def speak_response(self, text):
        try:
            self.is_speaking = True
            self.signal_emitter.log_signal.emit(f"Speaking: {text}")
            
            def speak_thread():
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
            
            # Run speech in separate thread with timeout
            thread = threading.Thread(target=speak_thread)
            thread.start()
            thread.join(timeout=1.0)  # Timeout after 1 second
            
            if thread.is_alive():
                self.is_speaking = False
                return
                
        except Exception:
            self.is_speaking = False

    def process_interaction(self):
        start_time = time.time()
        
        # Speech recognition with timeout
        speech_future = self.executor.submit(self.capture_speech)
        try:
            speech_text = speech_future.result(timeout=1.0)
        except TimeoutError:
            return
            
        if speech_text:
            self.signal_emitter.log_signal.emit(f"Detected: {speech_text}")
            
            # Response generation with timeout
            response_future = self.executor.submit(self.generate_response, speech_text)
            try:
                response_text = response_future.result(timeout=1.0)
            except TimeoutError:
                response_text = "Processing took too long."
            
            self.signal_emitter.log_signal.emit(f"Generated: {response_text}")
            
            # Ensure we're within time budget
            elapsed = time.time() - start_time
            if elapsed < 3.0:
                self.speak_response(response_text)

    def robust_continuous_interaction(self):
        while not self.stop_event.is_set():
            try:
                if not self.is_speaking:
                    self.process_interaction()
                time.sleep(0.1)
            except Exception as e:
                self.signal_emitter.log_signal.emit(f"Error: {e}")
                time.sleep(0.1)

    def closeEvent(self, event):
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        self.video_capture.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    bot = SpeechToSpeechBot()
    bot.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()