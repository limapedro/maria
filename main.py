# Our main file.

import speech_recognition as sr

# Cria um reconhecedor
r = sr.Recognizer()

# Abrir o microfone para captura
with sr.Microphone() as source:
    while True:
        audio = r.listen(source) # Define microfone como fonte de áudio

        print(r.recognize_google(audio, language='pt'))
