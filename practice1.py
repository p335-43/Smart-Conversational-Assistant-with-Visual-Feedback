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
        
        # Configure voice properties for more natural speech
        self.engine.setProperty('rate', 200)  # Adjust speaking rate
        # voices = self.engine.getProperty('voices')
        # self.engine.setProperty('voice', voices[2].id)  # Select a different voice if available

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
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
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
        Generate more detailed and contextually rich responses
        Implements advanced response generation with enhanced parameters
        """
        if not input_text:
            return "I'm sorry, but I couldn't hear what you said. Could you please repeat?"

        # Enhance the prompt to encourage more detailed responses
        enhanced_prompt = f"Provide a comprehensive and detailed explanation about: {input_text}. Give an in-depth response that covers multiple aspects."

        try:
            # Tokenize input with more sophisticated generation parameters
            input_ids = self.tokenizer(enhanced_prompt, return_tensors="pt", max_length=512, truncation=True).input_ids

            # Generate a more elaborate response
            outputs = self.model.generate(
                input_ids, 
                max_length=300,  # Increased max length for more detailed responses
                min_length=100,  # Ensure a minimum response length
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,  # Enable sampling for more diverse responses
                top_k=50,  # Top-k sampling
                top_p=0.95,  # Nucleus sampling
                temperature=0.7,  # Control randomness of response
                repetition_penalty=1.2  # Reduce repetition
            )

            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return response
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I encountered an error while generating a response. Could you try again?"

    def speak_response(self, text):
        """
        Convert text response to speech with improved articulation
        Implements enhanced speech synthesis
        """
        try:
            # Split long responses into chunks to prevent speech engine issues
            chunk_size = 100
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            for chunk in text_chunks:
                self.engine.say(chunk)
                self.engine.runAndWait()
        except Exception as e:
            print(f"Speech synthesis error: {e}")

    def process_interaction(self):
        """
        Orchestrate the entire speech-to-speech interaction
        Implements enhanced processing with more robust timing
        """
        start_time = time.time()

        # Speech Capture
        speech_text = self.capture_speech()
        
        # Response Generation
        if speech_text:
            print(f"Detected speech: {speech_text}")
            response_text = self.generate_response(speech_text)
            print(f"Generated response: {response_text}")
            
            # Speech Output
            self.speak_response(response_text)

        # Ensure total processing time is monitored
        processing_time = time.time() - start_time
        print(f"Total interaction processing time: {processing_time:.2f} seconds")

    def run(self):
        """
        Main interaction loop with improved user experience
        """
        print("Enhanced Speech-to-Speech Bot Initialized. Press Ctrl+C to exit.")
        try:
            while True:
                self.process_interaction()
                time.sleep(1)  # Small pause between interactions
        except KeyboardInterrupt:
            print("\nBot shutting down.")

def main():
    bot = SpeechToSpeechBot()
    bot.run()

if __name__ == "__main__":
    main()