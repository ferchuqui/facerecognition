import cv2
import mediapipe as mp
import pickle

# Inicializar MediaPipe para Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Cargar la imagen desde un archivo
image_path = 'ruta/a/tu/imagen.jpg'  # Cambia esto a la ruta de tu imagen
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Procesar la imagen y detectar los puntos clave de la cara
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    face_landmarks = results.multi_face_landmarks[0]

    # Almacenar los puntos clave en una lista
    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

    # Guardar los puntos clave en un archivo
    user_id = "fernando"  # ID del usuario
    with open(f"{user_id}_face_data.pkl", "wb") as file:
        pickle.dump(landmarks, file)

    print(f"Datos faciales del usuario {user_id} almacenados.")
else:
    print("No se detectó ninguna cara en la imagen.")
