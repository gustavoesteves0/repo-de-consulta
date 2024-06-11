from flask import Flask, Response, jsonify, send_from_directory
import cv2
import threading
import time
import numpy as np
from collections import deque

app = Flask(__name__)

# Inicializar os classificadores de detecção de rostos
frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Verificar se os classificadores foram carregados corretamente
if frontal_face_cascade.empty():
    print("Erro: O classificador de detecção de rostos frontal não pôde ser carregado.")
if profile_face_cascade.empty():
    print("Erro: O classificador de detecção de rostos de perfil não pôde ser carregado.")

# Inicializar o objeto de captura de vídeo
cap = cv2.VideoCapture(0)
cap_lock = threading.Lock()

# Variáveis globais para cálculo de latência
last_frame_time = time.time()
latency_values = deque(maxlen=10)

@app.route("/")
def read_root():
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        return str(e), 500

@app.route("/stream")
def stream():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/latency")
def get_latency():
    global latency_values
    avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0
    return jsonify(latency=avg_latency)

def generate_video():
    global last_frame_time
    global latency_values

    while True:
        with cap_lock:
            ret, frame = cap.read()
        if not ret:
            break

        # Converter a imagem para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Verificar se os classificadores estão vazios
        if not frontal_face_cascade.empty():
            # Detectar rostos frontais na imagem
            frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # Desenhar retângulos em rostos frontais
            for (x, y, w, h) in frontal_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if not profile_face_cascade.empty():
            # Detectar rostos de perfil na imagem
            profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # Desenhar retângulos em rostos de perfil
            for (x, y, w, h) in profile_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Codificar o frame em formato JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Definir o delimitador de frames
        boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'

        # Calcular a latência
        current_time = time.time()
        frame_latency = (current_time - last_frame_time) * 1000  # Convert latency to milliseconds
        latency_values.append(frame_latency)

        # Enviar o frame com o delimitador
        yield boundary + frame_bytes + b'\r\n'

        # Atualizar o tempo do último frame
        last_frame_time = current_time

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
