import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe para Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Cargar la imagen del avatar con fondo transparente
avatar_path = '/mnt/data/A_friendly_cartoon_avatar_with_a_round_face,_large.png'  # Ruta al archivo del avatar
avatar_img = cv2.imread(avatar_path, cv2.IMREAD_UNCHANGED)
if avatar_img is None:
    print("No se pudo abrir la imagen del avatar. Verifica la ruta del archivo.")
    exit()

# Función para superponer una imagen sobre otra
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Recortes
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Superponer la imagen
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0

    img_crop[:] = alpha * img_overlay_crop[:, :, :3] + (1 - alpha) * img_crop

# Índices de puntos clave para los ojos en MediaPipe Face Mesh
LEFT_EYE_INDEXES = [33, 133]
RIGHT_EYE_INDEXES = [362, 263]

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

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Obtener los puntos clave de los ojos
        left_eye = face_landmarks[LEFT_EYE_INDEXES[0]]
        right_eye = face_landmarks[RIGHT_EYE_INDEXES[0]]

        # Calcular la posición del avatar basado en los ojos
        eye_left = (int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0]))
        eye_right = (int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0]))
        face_width = int(np.linalg.norm(np.array(eye_left) - np.array(eye_right)))

        # Redimensionar el avatar para que coincida con el ancho de la cara
        avatar_resized = cv2.resize(avatar_img, (face_width, face_width))

        # Superponer el avatar en la cara
        overlay_image_alpha(frame, avatar_resized[:, :, :3], (eye_left[0] - face_width // 2, eye_left[1] - face_width // 2), avatar_resized[:, :, 3])

    cv2.imshow('Avatar Reemplazo Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
