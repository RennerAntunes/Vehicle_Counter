import cv2
import numpy as np
from time import sleep, time

def get_center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

offset = 6
detec = []
cars = 0
min_size = 80
delay = 60

cap = cv2.VideoCapture("video.mp4")
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    sleep(1/delay)
    start = time()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5,5), 5)
    img_sub = subtractor.apply(blur)
    dilated = cv2.dilate(img_sub, np.ones((5,5)))
    filled = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((5,5)))

    coutours, _ = cv2.findContours(filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    end = time()
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

    for (c) in coutours:
        (x,y,w,h) = cv2.boundingRect(c)
        contour_validation = w > min_size and h > min_size
        if not contour_validation:
            continue
        
        if y > 400:
            roi = frame[y:y + h, x: x + w]
            cv2.imshow("The_last_ one", roi)

            cv2.rectangle(frame, (x,y), (x + w, y + h), (255,255,255), 3)
            center = get_center(x,y,w,h)
            detec.append(center)

        for (x,y) in detec:
            if y < (550 + offset) and y >= (550 - offset):
                cars += 1
                cv2.line(frame, (50, 550), (1200, 550), (0,0,219), 4)
                detec.remove((x,y))
            else:
                detec.remove((x,y))
                 

    cv2.putText(frame, fps_label, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (36,36,36), 3)
    cv2.putText(frame, "VEHICLES DETECTED:" +str(cars), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),5)
    cv2.putText(frame, "VEHICLES DETECTED:" +str(cars), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (77,255,255), 2)
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()