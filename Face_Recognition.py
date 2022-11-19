from mtcnn.mtcnn import MTCNN
import cv2
import face_recognition
import csv
import numpy as np
person_name=[]
encode_List=[]
# names = []
def load_info():
    with open('friendstrain.csv','r') as file:
        reader = csv.reader(file)
        size = len(list(reader))

    with open('friendstrain.csv','r') as file:

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
    # img = img_small
    faces = detect.detect_faces(img)
    # print(faces)
    cv2.resize
    encoding_list = []

    for face in faces:
        x,y,w,h = face['box']
        encode = face_recognition.face_encodings(img[int(y-h/2):int(y+3*h/2),int(x-w/2):int(x+3*w/2)]) # exracting face and finding encoding of that portion only 
        # cv2.imshow("test",img[int(y):int(y+h),int(x):int(x+w)])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x,y,w,h = x*size,y*size,w*size,h*size
        # cv2.circle(img_small,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),1)
        cv2.circle(img_small,(int(x),int(y)),20,(255,0,0))
        if len(encode)==0:
            continue
        
        matches = face_recognition.compare_faces(encode_List,encode[0]) # getting a boolean array with true at matchinfg face index and flase at non matching face index
        facedis = face_recognition.face_distance(encode_List,encode[0]) # getting face dis

        matchIndex = np.argmin(facedis)

        

        if matches[matchIndex]:
            name = person_name[matchIndex][0].upper()
            cv2.putText(img_small,name,(int(x)+12,int(y+h)+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            names.append(name)

    
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # converting image from BGR to RGB

    facelocs = face_recognition.face_locations(image) # getting all the faces location in the given imae/videoframe
    # print(facelocs)
    encoding_curr_face = face_recognition.face_encodings(image) # getting encodings of all the faces we find in that photo
    # print(encoding_curr_face)
     # list for storing names of person present in our photo
    for encodeface, faceloc in zip(encoding_curr_face,facelocs): # looping for all the faces present in our current photo

        matches = face_recognition.compare_faces(encode_List,encodeface) # this method compares the current face encodings with all the encodings and return a list of true and false , true for where the face matches and false where it don't
        facedis = face_recognition.face_distance(encode_List,encodeface) # this metods also compares the face encodigs of curr face with all the encodings but it return a difference of the encodings

        matchIndex = np.argmin(facedis) # now we get the min value's index from faceis which denotes that the curr face matches most with that face in the database

        y1,x2,y2,x1 = faceloc # getiing x1,x2,y1,y2 for making the rectangle around the face
        # x1,y1 = top left corner
        # x2,y2 = bottom right corner
        x1,y1,x2,y2 = x1*size,y1*size,x2*size,y2*size
        # cv2.imshow("test",img[y1:y2,x1:x2])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if person_name[matchIndex][0].upper() not in names:
            # print(matches[matchIndex])
            cv2.rectangle(img_small,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2) # making a rectanle around the face
            if matches[matchIndex]: # checking at the min index if the face matces by checkng true and false at that index, proceed if true
                name = person_name[matchIndex][0].upper() # getting name of the person
                print(name)
                cv2.rectangle(img_small,(int(x1),int(y2)),(int(x2),int(y2)+50),(0,255,0),cv2.FILLED) # making rectangle for printing the name of the perosn in the photo 
                cv2.putText(img_small,name,(int(x1)+12,int(y2)+40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) # putting name of the person in the photo
                
                names.append(name)# storing all the names of person persent in the photo in the name list
        
    return names,img_small

img = cv2.imread("Test\\friends4.webp")
names,img_small = recognize(img,0.6)
print("recongnition done")
print(names)
cv2.imshow('test',img_small)
cv2.waitKey(0)