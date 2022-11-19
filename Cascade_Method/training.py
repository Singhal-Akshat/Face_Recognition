import cv2
import os
import numpy as np
import csv
names = []
def detect_face(classifier,colored_img,scaleFactor = 1.1):
    
    #colored_img = cv2.resize(colored_img, (0,0), None, 0.4,0.4)
    # print(colored_img)
    gray = cv2.cvtColor(colored_img,cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, scaleFactor = scaleFactor, minNeighbors = 5)
    
    if(len(faces) == 0):
        return None, None
    
    (x,y,w,h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):

    
    dirs = os.listdir(data_folder_path)

    faces = []
    labels= []
    i=1
    for image in dirs:
        path = os.path.join(data_folder_path,image)
        label = image.split('.')[0]
        print(label)
        # names.append(label)
        img = cv2.imread(path)
        # print("First Image:")
        # print(img)inr
        face,rect = detect_face(cascade,img)
        faces.append(face)
        labels.append(i)
        i+=1
        cv2.imshow("Training ",face)
        cv2.waitKey(100)
        cv2.destroyAllWindows

    return faces,labels

cascade = cv2.CascadeClassifier('Cascade_Method\classifiers\haarcascade_frontalface_alt.xml')
faces,labels = prepare_training_data('Train')
# print(labels)
# print(faces)
# print(names)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces,np.array(labels))

# test_img = cv2.imread('Test\\testing.jpg')

def predict(test_img):
    img = test_img.copy()
    # img = cv2.resize(img,(0,0),None,1,1)
    face,rect = detect_face(cascade,img)
    label = face_recognizer.predict(face)
    (x,y,w,h) = rect
    # print(names[label[0]-1])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img,str(names[label[0]-1]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

    return img
cap = cv2.VideoCapture(0)
while True:
    
    source,test_img = cap.read()
    cv2.imshow("detection",predict(test_img))
    if(cv2.waitKey(10)=='q'):
        break

