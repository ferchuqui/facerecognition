import cv2
import mediapipe as mp
import pickle
import numpy as np

# Cargar los datos faciales del usuario
user_id = "fernando"
with open(f"{user_id}_face_data.pkl", "rb") as file:
    stored_landmarks = pickle.load(file)

# Inicializar MediaPipe para Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)

def compare_landmarks(stored, current):
    # Convertir las listas de puntos clave en arrays de numpy
    stored = np.array(stored)
    current = np.array(current)

    # Calcular la distancia euclidiana entre los puntos clave
    distance = np.linalg.norm(stored - current)

    # Definir un umbral para considerar la cara como coincidente
    threshold = 1.0  # Ajusta este valor según sea necesario

    return distance < threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder a la cámara.")
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y detectar los puntos clave de la cara
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Obtener los puntos clave de la imagen actual
        current_landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

        # Comparar los puntos clave actuales con los almacenados
        if compare_landmarks(stored_landmarks, current_landmarks):
            cv2.putText(frame, "Bienvenido, Fernando", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Usuario no reconocido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Desbloqueo facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
