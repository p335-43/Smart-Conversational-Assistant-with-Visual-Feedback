import asyncio
import sounddevice as sd
import numpy as np
import threading
from transformers import pipeline
import torch
import queue
import logging
from faster_whisper import WhisperModel
from TTS.api import TTS
import webrtcvad
from typing import Optional

class SpeechBot:
    def __init__(self, debug: bool = True):
        """
        Initialize SpeechBot with improved error handling and logging
        """
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug
        
        try:
            # More robust model initialization with explicit compute type
            self.whisper = WhisperModel(
                "base", 
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float32"  # Explicitly set compute type
            )
            self.logger.info("Whisper model initialized")
        
            # TTS with fallback
            try:
                self.tts = TTS("tts_models/en/ljspeech/fast_pitch")
            except Exception as e:
                self.logger.warning(f"Primary TTS model failed, attempting alternative: {e}")
                self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            
            # VAD with more robust initialization
            self.vad = webrtcvad.Vad(3)
            
            # LLM with device selection and error handling
            device = 0 if torch.cuda.is_available() else -1
            try:
                self.llm = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    device=device
                )
            except Exception as e:
                self.logger.error(f"LLM initialization failed: {e}")
                self.llm = None
            
            # Audio configuration
            self.sample_rate = 16000
            self.channels = 1
            
            # Threading and concurrency controls
            self.recording = threading.Event()
            self.processing = threading.Event()
            
            # Thread-safe queues with maxsize to prevent unbounded growth
            self.audio_queue = queue.Queue(maxsize=50)
            self.response_queue = queue.Queue(maxsize=10)
        
        except Exception as e:
            self.logger.critical(f"Critical initialization error: {e}")
            raise

    def cleanup(self):
        """Ensure proper resource cleanup"""
        self.logger.info("Performing cleanup...")
        self.recording.clear()
        self.processing.clear()
        
        # Attempt safe queue clearing
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

    async def run(self):
        """Main run method with basic flow"""
        try:
            self.recording.set()
            self.processing.set()
            
            self.logger.info("Speech Bot starting...")
            
            # Simulate a simple interaction for demonstration
            test_text = "Hello, how are you today?"
            
            # Transcribe (simulate)
            self.logger.info(f"Simulated transcription: {test_text}")
            
            # Generate response
            if self.llm:
                response = self.llm(
                    test_text,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7
                )[0]['generated_text']
                self.logger.info(f"Generated response: {response}")
                
                # Convert to speech (simulate)
                try:
                    self.tts.tts_to_file(text=response, file_path="response.wav")
                    self.logger.info("Response saved to response.wav")
                except Exception as e:
                    self.logger.error(f"TTS conversion error: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in run method: {e}")
        finally:
            self.cleanup()

async def main():
    """Main async entry point"""
    bot = SpeechBot(debug=True)
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")