import cv2
import numpy as np

frame=cv2.VideoCapture('example_01.mp4')

# object detector
object_detector=cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=60)


count_line_position=140

def centre_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect=[]
offset=3 #Allowable error between pixels
number_of_people=0

while True:
    isTrue,cap=frame.read()
    height,width,_=cap.shape


    #print(height,width)

    #Extract region of interest
    Region_of_interest=cap

    #object detection
    mask=object_detector.apply(Region_of_interest)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(cap,(2,count_line_position),(400,count_line_position),(0,255,0),3)

    for cnt in contours:
        # calculate the object area and remove small elements
        area=cv2.contourArea(cnt)
        if area>800:
          # cv2.drawContours(Region_of_interest,[cnt],-1,(0,0,255),2)
           x,y,w,h=cv2.boundingRect(cnt)

           cv2.rectangle(Region_of_interest,(x,y),(x+w,y+h),(0,255,0),3)
           center=centre_handle(x,y, w, h)
           detect.append(center)
           cv2.circle(Region_of_interest,center,4,(0,0,255),-1)

           for(x,y) in detect:
               if y>(count_line_position-offset) and y<(count_line_position+offset):
                   number_of_people=number_of_people+1
                   cv2.line(cap, (10, count_line_position), (400, count_line_position), (0, 0,255), 3)
                   detect.remove((x,y))
           cv2.putText(cap, 'Total people:'+str(number_of_people), (50, 70), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255), 2)

    cv2.imshow('VIDEO',cap)

    cv2.imshow('VIDEO2', mask)

    key=cv2.waitKey(30)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()

