import cv2
import sys

camera_id = 0
delay = 1
window_name = 'face_recog'

cap = cv2.VideoCapture(0)

HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv.CascadeClassifier(HAAR_FILE)

if not cap.isOpened():
    sys.exit()

while True:
    ret, frame = cap.read()
    if ret == false: continue 
    face = cascade.detectMultiScale(frame)

    #顔部を枠で囲む
    for x, y, w, h in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)