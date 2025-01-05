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
from typing import Optional, List
import scipy.io.wavfile as wavfile
import os

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
            # Whisper model initialization
            self.whisper = WhisperModel(
                "base", 
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float32"
            )
            self.logger.info("Whisper model initialized")
        
            # TTS with fallback
            try:
                self.tts = TTS("tts_models/en/ljspeech/fast_pitch")
            except Exception as e:
                self.logger.warning(f"Primary TTS model failed, attempting alternative: {e}")
                self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            
            # VAD initialization with specific frame duration
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
            
            # Enhanced Audio Configuration
            self.sample_rate = 16000  # 16kHz is standard for VAD
            self.channels = 1
            self.frame_duration = 30  # 30ms frame size recommended for VAD
            self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
            
            # Threading and concurrency controls
            self.recording = threading.Event()
            self.processing = threading.Event()
            
            # Thread-safe queues
            self.audio_queue = queue.Queue(maxsize=50)
            self.transcription_queue = queue.Queue(maxsize=10)
            self.response_queue = queue.Queue(maxsize=10)
            
            # Conversation context
            self.conversation_history = []
        
        except Exception as e:
            self.logger.critical(f"Critical initialization error: {e}")
            raise

    def audio_capture_callback(self, indata, frames, time, status):
        """
        Robust audio capture callback with improved error handling
        """
        if status:
            self.logger.warning(f"Audio capture status: {status}")
            return
        
        try:
            # Ensure mono channel and correct data type
            if indata.ndim > 1:
                audio_frame = indata[:, 0]  # Take first channel if stereo
            else:
                audio_frame = indata.flatten()
            
            # Convert to int16 for VAD (required by WebRTC VAD)
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            
            # Ensure correct frame size for VAD (multiples of 10, 20, or 30 ms)
            if len(audio_int16) == self.frame_size:
                try:
                    # Check if frame contains speech
                    is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
                    
                    if is_speech:
                        # Put speech frames in queue
                        self.audio_queue.put(audio_frame)
                        if self.debug_mode:
                            self.logger.debug(f"Speech frame captured: {len(audio_frame)} samples")
                except Exception as e:
                    self.logger.error(f"VAD processing error: {e}")
            else:
                self.logger.warning(f"Incorrect frame size: {len(audio_int16)} vs expected {self.frame_size}")
        
        except Exception as e:
            self.logger.error(f"Error while processing audio frame: {e}")

    def collect_audio_segment(self):
        """Collect audio frames from queue into a single numpy array"""
        audio_frames = []
        while not self.audio_queue.empty():
            audio_frames.append(self.audio_queue.get())
        return np.concatenate(audio_frames)

    def transcribe_audio(self, audio_segment):
        """Transcribe audio using Whisper"""
        try:
            # Save audio segment to temporary file for Whisper
            temp_wav_path = "temp_audio_segment.wav"
            wavfile.write(temp_wav_path, self.sample_rate, 
                          (audio_segment * 32767).astype(np.int16))
            
            # Transcribe
            segments, _ = self.whisper.transcribe(temp_wav_path, language='en')
            transcription = ' '.join([segment.text for segment in segments])
            
            # Remove temporary file
            os.remove(temp_wav_path)
            
            return transcription
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None

    def generate_response(self, transcription):
        """Generate response using language model"""
        if self.llm:
            try:
                # Add transcription to conversation history
                self.conversation_history.append(transcription)
                
                # Limit conversation history to last 5 interactions
                if len(self.conversation_history) > 5:
                    self.conversation_history = self.conversation_history[-5:]
                
                # Combine conversation history for context
                context = " ".join(self.conversation_history)
                
                # Generate response
                response = self.llm(context)[0]['generated_text']
                
                # Add response to conversation history
                self.conversation_history.append(response)
                
                return response
            except Exception as e:
                self.logger.error(f"Response generation error: {e}")
        return "I'm sorry, I couldn't process that."

    def speak_response(self, response):
        """Convert text response to speech"""
        try:
            # Temporary audio file for TTS output
            output_wav_path = "response.wav"
            self.tts.tts_to_file(text=response, file_path=output_wav_path)
            
            # Play the audio file
            data, fs = wavfile.read(output_wav_path)
            sd.play(data, fs)
            sd.wait()
            
            # Optional: Remove temporary file
            os.remove(output_wav_path)
        except Exception as e:
            self.logger.error(f"Speech generation error: {e}")

    async def run(self):
        """
        Enhanced run method with audio processing
        """
        try:
            self.recording.set()
            self.processing.set()
            
            self.logger.info("Speech Bot starting...")
            
            # Start audio stream 
            stream = sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                dtype='float32',
                blocksize=self.frame_size,
                callback=self.audio_capture_callback
            )
            
            with stream:
                while self.recording.is_set():
                    # Check if we have enough audio to transcribe
                    if self.audio_queue.qsize() > 10:  # Threshold for speech detection
                        audio_segment = self.collect_audio_segment()
                        transcription = self.transcribe_audio(audio_segment)
                        
                        if transcription and transcription.strip():
                            self.logger.info(f"Transcribed: {transcription}")
                            response = self.generate_response(transcription)
                            self.logger.info(f"Response: {response}")
                            
                            # Speak the response
                            self.speak_response(response)
                    
                    await asyncio.sleep(0.5)  # Prevent tight loop
        
        except Exception as e:
            self.logger.error(f"Error in run method: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Ensure proper resource cleanup"""
        self.logger.info("Performing cleanup...")
        self.recording.clear()
        self.processing.clear()
        
        # Clear queues safely
        for q in [self.audio_queue, self.transcription_queue, self.response_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

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