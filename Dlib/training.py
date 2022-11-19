import os 
import cv2
import face_recognition
import csv
import numpy as np
path = 'train'
images =[] # it will store all the images of the given folder
classnames = [] # denotes the name of the person 

myList = os.listdir(path)  # will return all the images path inside that folder

for image in myList:
    currImg = cv2.imread(f'{path}/{image}') # getting image one by one
    images.append(currImg) # storing image in a list
    classnames.append(os.path.splitext(image)[0]) # getting image's name or we can say person's name

encode_List= []  # will store encoding of the images
for image in images:

    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #coverting from BGR to RGB
    encode = face_recognition.face_encodings(img)[0] #getting encoding of face in image
    encode_List.append(encode) # appendin geach encoding to our list

with open('trained.csv','a',newline="") as file: # opening a .csv file for storing the encoding so that we don't have to find the encoding again and again
    writer= csv.writer(file) # making a writer object for writing in csv file
    for name,encode in zip(classnames,encode_List): # looping in classnames and encode list and storing taht in our trained.csv file for future use
        writer.writerow([name]) # storing name first
        writer.writerow(encode) # then stroing that person's face encodings