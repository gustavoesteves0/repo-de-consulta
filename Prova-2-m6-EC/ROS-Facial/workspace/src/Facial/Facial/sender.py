import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
from datetime import datetime, timezone

# Configuração das constantes da câmera
IM_WIDTH = 1280
IM_HEIGHT = 720

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, '/video_frames', 10)
        self.latency_publisher_ = self.create_publisher(String, '/latency', 60)
        self.timer = self.create_timer(0.1, self.timer_callback)  # Publish every 0.1 seconds (10 Hz)
        self.latency_timer = self.create_timer(0.1, self.latency_callback)  # Calculate latency every 0.1 seconds
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().error('Error - could not open video device.')
            rclpy.shutdown()
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
        actual_video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f'Actual video resolution: {actual_video_width:.0f}x{actual_video_height:.0f}')

        # Inicializar os classificadores de detecção de rostos
        self.frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        # Verificar se os classificadores foram carregados corretamente
        if self.frontal_face_cascade.empty():
            self.get_logger().error("Erro: O classificador de detecção de rostos frontal não pôde ser carregado.")
        if self.profile_face_cascade.empty():
            self.get_logger().error("Erro: O classificador de detecção de rostos de perfil não pôde ser carregado.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Converter a imagem para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostos frontais na imagem
            if not self.frontal_face_cascade.empty():
                frontal_faces = self.frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # Desenhar retângulos em rostos frontais
                for (x, y, w, h) in frontal_faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Detectar rostos de perfil na imagem
            if not self.profile_face_cascade.empty():
                profile_faces = self.profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # Desenhar retângulos em rostos de perfil
                for (x, y, w, h) in profile_faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            msg = CompressedImage()
            msg.format = "jpeg"
            msg.data = buffer.tobytes()
            self.publisher_.publish(msg)
        else:
            self.get_logger().error('Failed to read frame from camera.')

    def latency_callback(self):
        current_time = datetime.now(timezone.utc).isoformat()
        self.get_logger().info(f'Sending timestamp: {current_time}')
        # Publish current timestamp
        latency_msg = String()
        latency_msg.data = current_time
        self.latency_publisher_.publish(latency_msg)

def main(args=None):
    rclpy.init(args=args)
    webcam_publisher = WebcamPublisher()
    rclpy.spin(webcam_publisher)
    webcam_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
