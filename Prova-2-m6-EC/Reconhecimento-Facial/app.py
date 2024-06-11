from flask import Flask, Response, jsonify, send_from_directory
import cv2
import time
import threading

app = Flask(__name__)

# Inicializar os classificadores de detecção de rostos
frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

print(frontal_face_cascade)
print(profile_face_cascade)

# Inicializar o objeto de captura de vídeo
cap = cv2.VideoCapture(0)
cap_lock = threading.Lock()

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
    latency = estimate_latency()
    return jsonify(latency=latency)

def generate_video():
    while True:
        with cap_lock:
            ret, frame = cap.read()
        if not ret:
            break

        # Converter a imagem para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos frontais na imagem
        frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Detectar rostos de perfil na imagem
        profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Desenhar um retângulo ao redor de cada rosto detectado
        for (x, y, w, h) in frontal_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        for (x, y, w, h) in profile_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Codificar o frame em formato JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Definir o delimitador de frames
        boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'

        # Enviar o frame com o delimitador
        yield boundary + frame_bytes + b'\r\n'

def estimate_latency():
    start_time = time.time()
    with cap_lock:
        cap.read()
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert latency to milliseconds
    return latency

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
