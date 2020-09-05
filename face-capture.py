import numpy as np
import cv2
import pickle
import os
import time
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# make directory for the concerned person
print("Enter your name of the person: ")
dir_name = input()
# print("Enter the Identification number of the person")
# id_no = input()
directory = dir_name #+ "-" + dir_name
# Parent Directory path 
parent_dir = "images"
# Path 
path = os.path.join(parent_dir, directory) 
os.mkdir(path)

cap = cv2.VideoCapture(0)
t_end = time.time() + 20
img_count = 1
while time.time() < t_end:
    #capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = color_frame[y:y+h, x:x+h]
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h

        img_item = 'images/{}/my-image{}.png'.format(directory,img_count)
        cv2.imwrite(img_item, roi_gray)
        img_count += 1
        img_items = 'images/{}/my-image{}.png'.format(directory,img_count)
        cv2.imwrite(img_items, frame)
        img_count += 1
        #cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
    # img_item = 'images/{}/my-image{}.jpg'.format(directory,img_count)
    # cv2.imwrite(img_item, frame)
    # img_count += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
