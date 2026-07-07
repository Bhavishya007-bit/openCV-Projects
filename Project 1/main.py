import cv2

# Use OpenCV's built-in cascade path — avoids relative path issues
face_cascade = cv2.CascadeClassifier("Project 1\\haarcascade_frontalface_default.xml")

# Fail fast with a clear message if the cascade didn't load
if face_cascade.empty():
    raise IOError("Failed to load cascade classifier XML file")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Failed to open webcam")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        '''
        detectMultiScale() - scan & detect faces 
        1.1 balance, not too slow, blind
        minNeighbors = 5
        minSize = ignores tiny false-positive detections
        '''

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show how many faces are currently detected
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Webcam Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()