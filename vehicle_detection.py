# -*- coding: utf-8 -*-


import cv2
import numpy as np

min_width=80 #minimum width of rectangle
min_height=80 #minimum height of rectangle
err=9 #offset
pos=650 #position of line
count=0
detect=[]

def get_center(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

cap=cv2.VideoCapture('video.mp4')
sub=cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret,frame=cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = sub.apply(blur)
    func=cv2.dilate(img_sub,np.ones((5,5)))
    #ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    di=cv2.morphologyEx (func, cv2. MORPH_CLOSE , kernel)
    di=cv2.morphologyEx (di, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(di,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(340,pos),(2000,pos),(255,127,0),3)
    for(i,j) in enumerate(contour):
        (x,y,w,h)=cv2.boundingRect(j)
        valid_contour=(w>=min_width)and(h>=min_height)
        if not valid_contour:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        center=get_center(x, y, w, h)
        detect.append(center)
        cv2.circle(frame,center,4,(0, 0,255),-1)
        
        for (x,y) in detect:
            if y<(pos+err) and y>(pos-err):
                count+=1
                cv2.line(frame,(340,pos),(2000,pos),(0,127,255),3)
                detect.remove((x,y))
                print("vehicle is detected : "+str(count))
        
    
    cv2.putText(frame,"VEHICLE COUNT : "+str(count),(450, 70),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255),5)
    cv2.imshow("Video Original",frame)
    cv2.imshow("Detectar",di)
    
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
    
        
        

