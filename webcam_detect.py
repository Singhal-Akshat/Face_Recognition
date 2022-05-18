import cv2
import numpy as np
import face_recognition
import os
import csv
# path = 'webcam'
# images =[]
# classnames = []

# myList = os.listdir(path)

# for image in myList:
#     currImg = cv2.imread(f'{path}/{image}')
#     images.append(currImg)
#     classnames.append(os.path.splitext(image)[0])

# encode_List= []
# for image in images:

#     img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     encode = face_recognition.face_encodings(img)[0]
#     encode_List.append(encode)

# print(encode_List)
# exit(0)

person_name=[]
encode_List=[]

with open('trained.csv','r') as file:
    reader = csv.reader(file)
    size = len(list(reader))

with open('trained.csv','r') as file:

    reader = csv.reader(file)
    i=0
    while i<size:
        print("i is: " + str(i))
        i+=2

        person_name.append(next(reader))
        encode_List.append(np.array(next(reader),float))

print(person_name[1][0])

cap = cv2.VideoCapture(0)
# url ='http://192.168.1.50:8080/video'
# cap = cv2.VideoCapture(url)
while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesloc = face_recognition.face_locations(imgs)
    encodecurrframe = face_recognition.face_encodings(imgs,facesloc)

    for encodeFace,faceLoc in zip(encodecurrframe,facesloc):
        matches = face_recognition.compare_faces(encode_List,encodeFace) # will return a list
        facedis = face_recognition.face_distance(encode_List,encodeFace) #will return a list

        matchIndex = np.argmin(facedis)

        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
        print(faceLoc)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #cv2.rectangle(imgS,(x1,y2-55),(x2,y2),(0,255,0),cv2.BORDER_TRANSPARENT)
        if matches[matchIndex]:
            name = person_name[matchIndex][0].upper()
            print(name)
            cv2.putText(img,name,(x1+12,y2-12),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        
        else:
            cv2.putText(img,"Not Known",(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        

    cv2.imshow('webcam',img)

    cv2.waitKey(1)



