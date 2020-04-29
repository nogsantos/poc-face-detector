import cv2
import numpy as np
from mtcnn import MTCNN


detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = detector.detect_faces(frame)

    for face in faces:

        x1, y1, w, h = face["box"]
        x2, y2 = x1 + w, y1 + h
        roi = frame[y1:y2, x1:x2]
        color = (255, 255, 255)
        label = "FACE"

        if not np.sum([roi]) == 0:
            roi = roi.astype("float") / 255.0
            label_position = (x1, y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    cv2.imshow("FACE DETECTOR", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
