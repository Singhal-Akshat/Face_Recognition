import cv2
import matplotlib.pyplot as plt

def detect_faces(classifier,colored_img,scaleFactor = 1.1):
    
    colored_img = cv2.resize(colored_img, (0,0), None, 0.4,0.4)
    gray = cv2.cvtColor(colored_img,cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, scaleFactor = scaleFactor, minNeighbors = 5)
    # i=255
    # ch ='a'
    for (x,y,w,h) in faces:
        cv2.rectangle(colored_img,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.putText(colored_img,ch,(x+4,y+4+h),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
        # i=i-10
        # print(x,y,w,h)
        # print(ch)
        # ch+='a'
    print("Faces found:  " + str(len(faces)))
    return colored_img

test = cv2.imread("Train\\Steve Jobs.jpg")

cascade = cv2.CascadeClassifier('Cascade_Method\classifiers\haarcascade_frontalface_alt.xml')
cv2.imshow("Final",detect_faces(cascade,test,1.1))
cv2.waitKey(0) 