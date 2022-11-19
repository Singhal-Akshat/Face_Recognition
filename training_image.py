import os
import cv2
import face_recognition
import csv

image = cv2.imread("celeb_train\\Rahul_gandhi.jpg")

img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
encode = face_recognition.face_encodings(img)[0]

with open('trained.csv','a',newline="") as file:
    writer= csv.writer(file)
    writer.writerow(["Rahul Gandhi"])
    writer.writerow(encode)