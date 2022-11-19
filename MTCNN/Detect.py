from mtcnn.mtcnn import MTCNN # importing MTCNN 0
import cv2

model = MTCNN() # creating a object of MTCNN
img = cv2.imread("Test\\all.jpg")

faces =  model.detect_faces(img) # caling detect_faces method of MTCNN object to get info and location of faces present in the given input
# img = cv2.resize(img,(0,0),None,0.2,0.2) # resizinf for display purposes
for face in faces:
    print(face)
    x,y,w,h = face['box']
    # x,y,w,h = x*0.2,y*0.2,w*0.2,h*0.2
    
    cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),1)# putiing rectangle around the face
    # print(face['box'])
    for key,value in face['keypoints'].items(): # getting location of eyes,nose and mouth
        x,y= value
        # x,y = x*0.2,y*0.2
        cv2.circle(img,(int(x),int(y)), 6,(0,0,255),3) # crearing a circle to show the identified area

cv2.imshow("image",img)
cv2.waitKey(0)