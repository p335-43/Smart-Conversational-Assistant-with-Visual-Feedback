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
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    vad_frame_ms: int = 30
    silence_threshold: float = 0.1
    silence_chunks: int = 15

class SpeechBot:
    def __init__(self):
        # Initialize speech recognition (Whisper)
        self.whisper = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize TTS
        self.tts = TTS("tts_models/en/ljspeech/fast_pitch")
        
        # Initialize VAD for real-time speech detection
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        
        # Initialize audio config
        self.config = AudioConfig()
        
        # Initialize audio queues
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize LLM pipeline
        self.llm = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Threading events
        self.recording = threading.Event()
        self.processing = threading.Event()

    async def process_audio_stream(self):
        """Process incoming audio stream with VAD"""
        buffer = []
        silence_counter = 0
        
        while self.recording.is_set():
            if self.audio_queue.empty():
                await asyncio.sleep(0.01)
                continue
                
            audio_chunk = self.audio_queue.get()
            is_speech = self.vad.is_speech(
                audio_chunk.tobytes(),
                self.config.sample_rate,
                self.config.vad_frame_ms
            )
            
            if is_speech:
                buffer.append(audio_chunk)
                silence_counter = 0
            else:
                silence_counter += 1
                
            if silence_counter >= self.config.silence_chunks and buffer:
                await self.process_speech(np.concatenate(buffer))
                buffer = []
                silence_counter = 0

    async def process_speech(self, audio_data: np.ndarray):
        """Process speech with Whisper and LLM"""
        # Convert audio to text
        result = self.whisper.transcribe(audio_data)
        text = result[0].text
        
        # Generate response with LLM
        response = self.llm(
            text,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7
        )[0]['generated_text']
        
        # Generate speech from response
        speech = self.tts.tts(response)
        self.response_queue.put(speech)

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio callback error: {status}")
        self.audio_queue.put(indata.copy())

    async def run(self):
        """Main run loop"""
        self.recording.set()
        self.processing.set()
        
        # Start audio stream
        with sd.InputStream(
            callback=self.audio_callback,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size
        ):
            # Start processing loop
            await self.process_audio_stream()
            
            # Play responses
            while self.processing.is_set():
                if not self.response_queue.empty():
                    response_audio = self.response_queue.get()
                    sd.play(response_audio, self.config.sample_rate)
                    sd.wait()
                await asyncio.sleep(0.1)

    def stop(self):
        """Stop the bot"""
        self.recording.clear()
        self.processing.clear()

if __name__ == "__main__":
    bot = SpeechBot()
    asyncio.run(bot.run())