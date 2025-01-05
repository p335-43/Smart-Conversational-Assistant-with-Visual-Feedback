

# Smart Conversational Assistant with Visual Feedback  

## Overview  
The **Smart Conversational Assistant with Visual Feedback** is a cutting-edge desktop application that combines speech recognition, AI-powered response generation, and live video integration. It provides an engaging and intuitive platform for seamless human-computer interaction, designed for applications in customer service, education, and assistive technologies.  

---

## Features  
- **Real-Time Speech Recognition**  
   - Captures and transcribes user speech accurately using the SpeechRecognition library.  

- **AI-Powered Response Generation**  
   - Generates thoughtful and context-aware responses leveraging the FLAN-T5 model.  

- **Speech Synthesis**  
   - Delivers natural and audible responses through the pyttsx3 engine.  

- **Live Video Feed**  
   - Displays real-time video from the system's camera using OpenCV and PyQt5 for an engaging interface.  

- **User-Friendly GUI**  
   - A clean and intuitive interface built with PyQt5 for effortless interaction.  

---

## Installation  

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/p335-43/Smart-Conversational-Assistant-with-Visual-Feedback.git
   cd smart-conversational-assistant
   ```

2. **Install Dependencies:**  
   Ensure you have Python 3.8+ installed. Install the required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**  
   ```bash
   python main.py
   ```

---

## Technologies Used  
- **Speech Recognition:** `speech_recognition`  
- **AI Language Model:** `transformers` (FLAN-T5)  
- **Text-to-Speech (TTS):** `pyttsx3`  
- **Video Processing:** `OpenCV`  
- **Graphical User Interface (GUI):** `PyQt5`  

---

## How It Works  
1. The system captures speech using the microphone and processes it with the SpeechRecognition library.  
2. The captured speech is passed to the FLAN-T5 model, which generates a context-aware response.  
3. The response is synthesized into speech using the pyttsx3 TTS engine.  
4. A live video feed is displayed in the application interface, enhancing user engagement.  

---

## Future Enhancements  
- Add support for multiple languages.  
- Integrate emotion detection using facial analysis.  
- Optimize for mobile devices or web platforms.  

