import cv2
import mediapipe as mp

# Inicializar los módulos de MediaPipe para manos y dibujo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Crear una instancia de la solución de manos
hands = mp_hands.Hands()

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder a la cámara.")
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y detectar manos
    results = hands.process(image_rgb)

    # Dibujar las marcas de las manos en la imagen
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar la imagen con las manos detectadas
    cv2.imshow('Manos detectadas', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()