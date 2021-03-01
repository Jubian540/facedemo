#!/usr/bin/env python3

import cv2

camera = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

def main():
    while True:
        ret,frame = camera.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
    pass

if __name__ == '__main__':
    main()
    camera.release()
