import os
import shutil
import cv2
from detector_image import recognize
src = "Test"
dest ="Segregate"

all = os.listdir(src)

for img in all:
    im = cv2.imread(os.path.join(src,img))
    names,i = recognize(im,1)

    for name in names:
        if(not os.path.isdir(os.path.join(dest,name))):
            os.makedirs(os.path.join(dest,name))
        
        print("copying " + img + " to "+ name)
        shutil.copy(os.path.join(src,img),os.path.join(dest,name))
