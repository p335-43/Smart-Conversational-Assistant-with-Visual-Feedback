import asyncio
import sounddevice as sd
import numpy as np
import threading
from transformers import pipeline
import torch
import queue
import wave
from faster_whisper import WhisperModel
from TTS.api import TTS
import webrtcvad
import contextlib
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AudioConfig:
    sample_rate: int = 16000  # Must be one of: 8000, 16000, 32000, 48000
    channels: int = 1
    vad_frame_ms: int = 30    # Must be 10, 20, or 30
    chunk_size: int = int(16000 * 30 / 1000)  # Exactly 30ms worth of samples
    silence_threshold: float = 0.1
    silence_chunks: int = 15
    max_speech_duration: float = 30.0  # Maximum duration of speech in seconds

    def __post_init__(self):
        # Validate configuration
        assert self.sample_rate in [8000, 16000, 32000, 48000], "Invalid sample rate"
        assert self.vad_frame_ms in [10, 20, 30], "Invalid VAD frame duration"
        assert self.channels == 1, "Only mono audio is supported"

class SpeechBot:
    def __init__(self, debug: bool = False):
        """
        Initialize SpeechBot with optional debug mode
        
        :param debug: Enable detailed debug logging
        """
        self.debug_mode = debug
        
        def log(message):
            """Internal logging method"""
            if self.debug_mode:
                print(f"[DEBUG] {message}")
        
        log("Initializing Speech Bot components...")
        
        try:
            # Initialize speech recognition (Whisper)
            self.whisper = WhisperModel(
                "base", 
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            log("Whisper model initialized")
        
            # Initialize TTS
            self.tts = TTS("tts_models/en/ljspeech/fast_pitch")
            log("TTS model initialized")
        
            # Initialize VAD for real-time speech detection
            self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
            log("VAD initialized")
        
            # Initialize audio config
            self.config = AudioConfig()
        
            # Initialize audio queues
            self.audio_queue = queue.Queue()
            self.response_queue = queue.Queue()
        
            # Initialize LLM pipeline
            log("Initializing LLM...")
            self.llm = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            log("LLM initialized")
        
            # Threading events
            self.recording = threading.Event()
            self.processing = threading.Event()
        
            log("Speech Bot initialization complete")
        
        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def debug_log(self, message: str):
        """Conditional debug logging"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")

    def prepare_audio_for_vad(self, audio_data: np.ndarray) -> Optional[bytes]:
        """Convert audio data to format required by WebRTC VAD"""
        try:
            # Ensure input is a numpy array
            if not isinstance(audio_data, np.ndarray):
                self.debug_log(f"Invalid audio data type: {type(audio_data)}")
                return None
            
            # Flatten multi-channel audio
            if len(audio_data.shape) > 1:
                if audio_data.shape[1] > 1:
                    audio_data = audio_data.mean(axis=1)
                audio_data = audio_data.flatten()
            
            # Ensure audio data is float type and normalize
            audio_data = audio_data.astype(np.float32)
            
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normalize to prevent potential overflow
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Ensure exact frame size for VAD
            required_samples = int(self.config.sample_rate * self.config.vad_frame_ms / 1000)
            
            # Add more robust padding/truncation
            if len(audio_data) != required_samples:
                if len(audio_data) < required_samples:
                    # Zero-pad if too short
                    audio_data = np.pad(audio_data, (0, required_samples - len(audio_data)), 
                                        mode='constant', constant_values=0)
                else:
                    # Trim if too long
                    audio_data = audio_data[:required_samples]
            
            # Final validation
            if len(audio_data) != required_samples:
                self.debug_log(f"Audio frame size incorrect. Expected {required_samples}, got {len(audio_data)}")
                return None
            
            return audio_data.tobytes()
        
        except Exception as e:
            self.debug_log(f"Detailed VAD preparation error: {str(e)}")
            self.debug_log(f"Audio data details: "
                           f"shape={audio_data.shape if isinstance(audio_data, np.ndarray) else 'N/A'}, "
                           f"type={type(audio_data)}, "
                           f"dtype={audio_data.dtype if isinstance(audio_data, np.ndarray) else 'N/A'}")
            return None

    async def process_audio_stream(self):
        """Process incoming audio stream with VAD"""
        buffer = []
        silence_counter = 0
        total_duration = 0
        
        self.debug_log("Starting audio stream processing...")
        
        while self.recording.is_set():
            try:
                # Use a shorter timeout to prevent blocking
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                if audio_chunk is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Ensure the chunk is a numpy array
                if not isinstance(audio_chunk, np.ndarray):
                    self.debug_log(f"Unexpected audio chunk type: {type(audio_chunk)}")
                    await asyncio.sleep(0.1)
                    continue
                
                chunk_duration = len(audio_chunk) / self.config.sample_rate
                
                # More robust VAD audio preparation
                vad_audio = self.prepare_audio_for_vad(audio_chunk)
                
                if vad_audio is None:
                    self.debug_log("Skipping frame due to VAD preparation failure")
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    is_speech = self.vad.is_speech(
                        vad_audio,
                        sample_rate=self.config.sample_rate
                    )
                except Exception as e:
                    self.debug_log(f"VAD processing error: {str(e)}")
                    await asyncio.sleep(0.1)
                    continue
                
                if is_speech:
                    buffer.append(audio_chunk)
                    silence_counter = 0
                    total_duration += chunk_duration
                    
                    if total_duration >= self.config.max_speech_duration:
                        if buffer:
                            self.debug_log("Max duration reached, processing speech...")
                            await self.process_speech(np.concatenate(buffer))
                            buffer = []
                            total_duration = 0
                else:
                    silence_counter += 1
                    
                if silence_counter >= self.config.silence_chunks and buffer:
                    self.debug_log("Silence detected, processing speech...")
                    await self.process_speech(np.concatenate(buffer))
                    buffer = []
                    silence_counter = 0
                    total_duration = 0
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.debug_log(f"Unexpected error in audio processing: {str(e)}")
                await asyncio.sleep(0.1)

    async def process_speech(self, audio_data: np.ndarray):
        """Process speech with Whisper and LLM"""
        try:
            self.debug_log("Converting speech to text...")
            # Convert audio to text
            result = self.whisper.transcribe(audio_data)
            text = result[0].text
            
            if not text.strip():
                self.debug_log("No speech detected in audio")
                return
                
            self.debug_log(f"Transcribed text: {text}")
            
            # Generate response with LLM
            self.debug_log("Generating response...")
            response = self.llm(
                text,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7
            )[0]['generated_text']
            
            self.debug_log(f"Generated response: {response}")
            
            # Generate speech from response
            self.debug_log("Converting response to speech...")
            speech = self.tts.tts(response)
            self.response_queue.put(speech)
            self.debug_log("Response ready for playback")
            
        except Exception as e:
            self.debug_log(f"Error in speech processing: {str(e)}")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            self.debug_log(f"Audio callback status: {status}")
            return
        
        try:
            # Ensure indata is a numpy array and has correct dimensions
            if isinstance(indata, np.ndarray):
                self.audio_queue.put(indata.copy())
            else:
                self.debug_log(f"Unexpected audio input type: {type(indata)}")
        except Exception as e:
            self.debug_log(f"Error in audio callback: {str(e)}")

    async def run(self, debug: bool = False):
        """Main run loop"""
        try:
            self.debug_mode = debug
            self.debug_log("Initializing Speech Bot...")
            self.recording.set()
            self.processing.set()
            
            self.debug_log("Starting audio stream...")
            with sd.InputStream(
                callback=self.audio_callback,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                dtype=np.float32,
                latency='low',
                device=None  # Use default input device
            ):
                self.debug_log("Audio stream started. Listening...")
                
                # Create tasks for audio processing and response playback
                audio_task = asyncio.create_task(self.process_audio_stream())
                
                # Response playback loop
                while self.processing.is_set():
                    if not self.response_queue.empty():
                        response_audio = self.response_queue.get()
                        self.debug_log("Playing response...")
                        sd.play(response_audio, self.config.sample_rate)
                        sd.wait()
                    await asyncio.sleep(0.1)
                
                # Wait for audio processing to complete
                await audio_task
                    
        except Exception as e:
            self.debug_log(f"Error in main loop: {str(e)}")
            self.stop()
        finally:
            self.debug_log("Shutting down Speech Bot...")

    def stop(self):
        """Stop the bot"""
        self.debug_log("Stopping Speech Bot...")
        self.recording.clear()
        self.processing.clear()

async def handle_shutdown(bot):
    """Handle graceful shutdown on keyboard interrupt"""
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        bot.stop()

if __name__ == "__main__":
    try:
        print("Starting Speech Bot application...")
        
        # Create and run the bot with debug mode
        bot = SpeechBot()
        asyncio.run(bot.run(debug=True))
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        if 'bot' in locals():
            bot.stop()