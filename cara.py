import cv2
import mediapipe as mp

# Inicializar los módulos de MediaPipe para rostros y dibujo
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Crear una instancia de la solución de detección de rostros
face_detection = mp_face_detection.FaceDetection()

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder a la cámara.")
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y detectar rostros
    results = face_detection.process(image_rgb)

    # Dibujar las marcas de los rostros en la imagen
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Mostrar la imagen con los rostros detectados
    cv2.imshow('Rostros detectados', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
