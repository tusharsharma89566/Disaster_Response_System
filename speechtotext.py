import speech_recognition as sr

def convert_speech_to_text():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Please speak something...")
        
        try:
            audio = recognizer.listen(source, timeout=5)
            print("Processing speech...")
            
            text = recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            return "No speech detected"
        except sr.RequestError:
            return "Could not connect to speech recognition service"
        except sr.UnknownValueError:
            return "Could not understand the audio"

def main():
    result = convert_speech_to_text()
    print(f"Recognized text: {result}")

if __name__ == "__main__":
    main()