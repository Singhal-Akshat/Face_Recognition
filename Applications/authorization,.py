from detector_image import recognize
# from Face_Recognition import recognize
import cv2
import csv


cap = cv2.VideoCapture(0)
while True:
    source, img = cap.read()
    name,imgo = recognize(img,1)
    print(name)
    cv2.imshow('webcam',imgo)
    cv2.waitKey(0)
    if(len(name)==1):
        if name[0] == "AKSHAT":
            print("Authorization successful")
            exit(1)