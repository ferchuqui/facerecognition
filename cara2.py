import cv2
import mediapipe as mp

# Inicializar los módulos de MediaPipe para Face Mesh y dibujo
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Crear una instancia de la solución de Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder a la cámara.")
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y detectar los puntos clave de la cara
    results = face_mesh.process(image_rgb)

    # Dibujar los puntos clave de la cara en la imagen
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
            )

    # Mostrar la imagen con los puntos clave de la cara detectados
    cv2.imshow('Puntos clave de la cara detectados', frame)

    # Salir del bucle si se presiona  la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
