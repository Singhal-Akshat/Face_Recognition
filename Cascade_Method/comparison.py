from detect import detect_face
import cv2
import time
test = cv2.imread("Test\\testing.jpg")

haar_cascade = cv2.CascadeClassifier('Cascade_Method\classifiers\haarcascade_frontalface_alt.xml')
t1 = time.time()


haar = detect_face(haar_cascade,test)

t2 = time.time()

print(t2-t1)

lbp_cascade = cv2.CascadeClassifier('Cascade_Method\classifiers\lbpcascade_frontalface_improved.xml')
t1 = time.time()

lbp = detect_face(lbp_cascade,test,1.01)
t2 = time.time()

print(t2-t1)
cv2.imshow("lbp",lbp)
cv2.waitKey(0)
