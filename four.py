import speech_recognition as sr
import torch
import transformers
import pyttsx3
import threading
import queue
import time

class SpeechToSpeechBot:
    def __init__(self):

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-medium')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-medium')
        self.engine = pyttsx3.init()

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_lock = threading.Lock()

    def capture_speech(self):

        with self.microphone as source:

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

        if not input_text:
            return "I didn't catch that. Could you repeat?"


        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(
            inputs, 
            max_length=50,  
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def speak_response(self, text):
      
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech synthesis error: {e}")

    def process_interaction(self):
    
        start_time = time.time()

        speech_text = self.capture_speech()
        
        if speech_text:
            response_text = self.generate_response(speech_text)
            
            
            self.speak_response(response_text)

        processing_time = time.time() - start_time
        if processing_time > 3:
            print(f"Warning: Processing took {processing_time:.2f} seconds")

    def run(self):
    
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