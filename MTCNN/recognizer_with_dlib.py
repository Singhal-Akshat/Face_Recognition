from mtcnn.mtcnn import MTCNN
import cv2
import face_recognition
import csv
import numpy as np
person_name=[]
encode_List=[]

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

# till now ew have retained all the encoding and names and now we can recognize people with it
detect = MTCNN()
img = cv2.imread("Test\\all.jpg")
img_small = cv2.resize(img,(0,0),None,0.4,0.4)
# img_small = img
faces = detect.detect_faces(img)
print("Faceslocation are : ")
print(faces)
encoding_list = []
for face in faces:
    x,y,w,h = face['box']
    encode = face_recognition.face_encodings(img[int(y-h/2):int(y+3*h/2),int(x-w/2):int(x+3*w/2)]) # exractinf face and finding encoding of that portion only 
    cv2.imshow("test",img[int(y-h/2):int(y+3*h/2),int(x-20):int(x+w+20)])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x,y,w,h = x*0.4,y*0.4,w*0.4,h*0.4
    cv2.rectangle(img_small,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),1)
    if len(encode)==0:
        continue
    # exit(1)
    # encoding_list.append(encode)
    
    matches = face_recognition.compare_faces(encode_List,encode[0]) # getting a boolean array with true at matchinfg face index and flase at non matching face index
    facedis = face_recognition.face_distance(encode_List,encode[0]) # getting face dis

    matchIndex = np.argmin(facedis)

    
    # cv2.rectangle(img_small,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),1)

    if matches[matchIndex]:
        name = person_name[matchIndex][0].upper()
        print(name)
        cv2.putText(img_small,name,(int(x)+12,int(y+h)+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

print("recongnition done")
cv2.imshow('test',img_small)
cv2.waitKey(0)
# print(encoding_list)