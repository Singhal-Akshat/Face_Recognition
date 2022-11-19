from detector_image import recognize
# from Face_Recognition import recognize
import cv2
import csv
import time

t = time.localtime()
# currtime = time.strftime("%H:%mM:%S",t)
# # for webcam
# cap = cv2.VideoCapture(0)
# while True:
#     source, img = cap.read()
#     name,imgo = recognize(img,1)
#     print(name)
#     cv2.imshow('webcam',imgo)
#     cv2.waitKey(20)
#     names= []

# from photo
img= cv2.imread("Test\\all.jpg")
name,imgo = recognize(img,1)
print(name)
cv2.imshow('webcam',img)
cv2.waitKey(20)
names= []
with open("attendance.csv",'r') as file:
    reader = csv.reader(file)

    for nme in reader:
        if not len(nme) == 0:
            names.append(nme[0])

with open("attendance.csv",'a',newline="") as file:
    writer = csv.writer(file)
    print(name)
    for nm in name:
        print(nm)
        if nm not in names:
            print("inside")
            t = time.localtime()
            currtime = time.strftime("%H:%mM:%S",t)
            writer.writerow([nm,currtime])