import cv2
import numpy as np
import face_recognition
import csv
person_name=[] # list to retrieve names from our trained.csv file or (any database)
encode_List=[] # list to get encodings which we have stored earlier

def info():
    with open('trained.csv','r') as file: # opening the trained.csv file to get back the results
        reader = csv.reader(file) # making a reader object
        size = len(list(reader)) # getting it's size

    with open('trained.csv','r') as file:

        reader = csv.reader(file)
        i=0
        while i<size: # looping till the end of file
            i+=2 # incrementing by 2 each time so that we get current name and thier encodings

            person_name.append(next(reader)) # appeding person's name in our list
            encode_List.append(np.array(next(reader),float)) # appendin that person's encodings in our list

# now we have retained all the information which we stored earlier now we will indentify that the current face belongs to which person

# test_image = cv2.imread('Test\\Elon_jeff.jpg') # loading the image where we want to apply the face recognition 
# imgS = cv2.resize(test_image,(0,0),None,0.5,0.5) # resizing 
def recognize(test_image,size):
    imgS = cv2.resize(test_image,(0,0),None,size,size) 

    img = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)  # converting image from BGR to RGB

    facelocs = face_recognition.face_locations(img) # getting all the faces location in the given imae/videoframe
    encoding_curr_face = face_recognition.face_encodings(img) # getting encodings of all the faces we find in that photo

    names = [] # list for storing names of person present in our photo
    for encodeface, faceloc in zip(encoding_curr_face,facelocs): # looping for all the faces present in our current photo

        matches = face_recognition.compare_faces(encode_List,encodeface) # this method compares the current face encodings with all the encodings and return a list of true and false , true for where the face matches and false where it don't
        facedis = face_recognition.face_distance(encode_List,encodeface) # this metods also compares the face encodigs of curr face with all the encodings but it return a difference of the encodings

        matchIndex = np.argmin(facedis) # now we get the min value's index from faceis which denotes that the curr face matches most with that face in the database

        y1,x2,y2,x1 = faceloc # getiing x1,x2,y1,y2 for making the rectangle around the face
        # x1,y1 = top left corner
        # x2,y2 = bottom right corner
        x1,y1,x2,y2 = x1*size,y1*size,x2*size,y2*size
        cv2.rectangle(imgS,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2) # making a rectanle around the face
        

        if matches[matchIndex]: # checking at the min index if the face matces by checkng true and false at that index, proceed if true
            name = person_name[matchIndex][0].upper() # getting name of the person
            cv2.rectangle(imgS,(int(x1),int(y2)),(int(x2),int(y2)+50),(0,255,0),cv2.FILLED) # making rectangle for printing the name of the perosn in the photo 
            cv2.putText(imgS,name,(int(x1)+12,int(y2)+40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) # putting name of the person in the photo
            
            names.append(name)# storing all the names of person persent in the photo in the name list
        else:
            # if the person is not present then showing that we don't recognnize that person and we need data to identify this person in future
            cv2.rectangle(imgS,(int(x1),int(y2)),(int(x2),int(y2)+50),(0,255,0),cv2.FILLED) 
            cv2.putText(imgS,"Not Known",(int(x1)+12,int(y2)+40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    return names,imgS

info()
# test_image = cv2.imread('Test\\all.jpg') # loading the image where we want to apply the face recognition 
# names,img = recognize(test_image,1)
# print(names)
# cv2.imshow('test',img) # showing the image with rectagles around the face and name in the bottom
# cv2.waitKey(0) # waiting till user presses a key 

