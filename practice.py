import speech_recognition as sr
import torch
import transformers
import pyttsx3
import threading
import queue
import time

class SpeechToSpeechBot:
    def __init__(self):
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Language Model Setup (FLAN-T5 Large)
        self.model = transformers.T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('google/flan-t5-large')

        # Text-to-Speech Setup
        self.engine = pyttsx3.init()

        # Processing Queue and Synchronization
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_lock = threading.Lock()

    def capture_speech(self):
        """
        Capture and transcribe speech input
        Implements noise reduction and adaptive recognition
        """
        with self.microphone as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=3)
                text = self.recognizer.recognize_google(audio)
                return text.strip()
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError:
                print("Speech recognition service error")
                return None

    def generate_response(self, input_text):
        """
        Generate contextually appropriate response using FLAN-T5
        Implements response generation with token limits
        """
        if not input_text:
            return "I didn't catch that. Could you repeat?"

        # Tokenize input and generate response
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids, 
            max_length=50,  # Limit response length
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def speak_response(self, text):
        """
        Convert text response to speech
        Implements natural-sounding speech generation
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech synthesis error: {e}")

    def process_interaction(self):
        """
        Orchestrate the entire speech-to-speech interaction
        Implements 3-second processing window
        """
        start_time = time.time()

        # Speech Capture
        speech_text = self.capture_speech()
        
        # Response Generation
        if speech_text:
            response_text = self.generate_response(speech_text)
            
            # Speech Output
            self.speak_response(response_text)

        # Ensure total processing time is within 3 seconds
        processing_time = time.time() - start_time
        if processing_time > 3:
            print(f"Warning: Processing took {processing_time:.2f} seconds")

    def run(self):
        """
        Main interaction loop
        """
        print("Speech-to-Speech Bot Initialized. Press Ctrl+C to exit.")
        try:
            while True:
                self.process_interaction()
        except KeyboardInterrupt:
            print("\nBot shutting down.")

def main():
    bot = SpeechToSpeechBot()
    bot.run()

if __name__ == "__main__":
    main()