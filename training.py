import os
import cv2
import face_recognition
import csv
import numpy as np
path = 'webcam'
images =[]
classnames = []

myList = os.listdir(path)

for image in myList:
    currImg = cv2.imread(f'{path}/{image}')
    images.append(currImg)
    classnames.append(os.path.splitext(image)[0])

encode_List= []
for image in images:

    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encode_List.append(encode)

#encode_List = encode_List.tolist()
print(encode_List)
with open('trained.csv','w',newline="") as file:
    writer= csv.writer(file)
    for name,encode in zip(classnames,encode_List):
        #encode = encode.tolist()
        #print(encode)
        writer.writerow([name])
        writer.writerow(encode)
# encode_list=[]
# with open('trained.csv','r') as file:
#     reader = csv.reader(file)
#     next(reader)
#     encode_list.append(np.array(next(reader),float))
#     next(reader)
#     encode_list.append(np.array(next(reader),float))

# print("Now printing")
# print(encode_list)