import cv2
import mediapipe as mp
import pickle

# Inicializar MediaPipe para Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)
user_id = "user_1"  # ID del usuario

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

        # Almacenar los puntos clave en un diccionario
        landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

        # Guardar los puntos clave en un archivo
        with open(f"{user_id}_face_data.pkl", "wb") as file:
            pickle.dump(landmarks, file)

        print(f"Datos faciales del usuario {user_id} almacenados.")
        break

    cv2.imshow('Captura de datos faciales', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
