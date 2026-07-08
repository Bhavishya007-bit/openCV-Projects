import cv2
import time
import numpy as np
import HandTrackingModule as htm

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cv2.circle(img, (x1,y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cTime = time.time()
    fps = 1/(cTime- pTime)
    pTime = cTime

    cv2.putText(img, f'FPS, {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3)
    cv2.imshow("Img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()