from mtcnn.mtcnn import MTCNN
import cv2
import face_recognition
import csv
import numpy as np
person_name=[]
encode_List=[]
# names = []
def load_info():
    with open('trained.csv','r') as file:
        reader = csv.reader(file)
        size = len(list(reader))

    with open('trained.csv','r') as file:

        reader = csv.reader(file)
        i=0
        while i<size:
            i+=2

            person_name.append(next(reader))
            encode_List.append(np.array(next(reader),float))

# till now we have retained all the encoding and names and now we can recognize people with it


def recognize(img,size):
    load_info()
    names= []
    detect = MTCNN()
    # img = cv2.imread("Test\\Family.jpg")

    img_small = cv2.resize(img,(0,0),None,size,size)
    faces = detect.detect_faces(img)


    encoding_list = []

    for face in faces:
        x,y,w,h = face['box']
        encode = face_recognition.face_encodings(img[y:y+h,x:x+w]) # exracting face and finding encoding of that portion only 

        if len(encode)==0:
            continue
        
        matches = face_recognition.compare_faces(encode_List,encode[0]) # getting a boolean array with true at matchinfg face index and flase at non matching face index
        facedis = face_recognition.face_distance(encode_List,encode[0]) # getting face dis

        matchIndex = np.argmin(facedis)

        x,y,w,h = x*size,y*size,w*size,h*size
        cv2.rectangle(img_small,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),1)

        if matches[matchIndex]:
            name = person_name[matchIndex][0].upper()
            cv2.putText(img_small,name,(int(x)+12,int(y+h)+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            names.append(name)

    return names,img_small

img = cv2.imread("Test\\family.jpg")
names,img_small = recognize(img,0.4)
print("recongnition done")
print(names)
cv2.imshow('test',img_small)
cv2.waitKey(0)