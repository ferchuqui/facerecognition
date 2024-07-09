import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/2.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mpFaceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mpFaceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime =cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)