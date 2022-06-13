import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
from message import send_message

path = 'student_images'
images = []
classNames = []
video=[]
mylist = os.listdir(path)
d={'anuhya':0,'madhu':0,'pavitra':0,'prerana':0}
sent = {'anuhya':0,'madhu':0,'pavitra':0,'prerana':0}
for file in mylist:
    curImg = cv2.imread(f'{path}/{file}')
    images.append(curImg)
    classNames.append(os.path.splitext(file)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

cap = cv2.VideoCapture('VID-20220607-WA0012.mp4') 
count =0
while True:
    count+=1
    if count%20 !=0:####################################### why
        continue
    # Capture frame-by-frame'
    ret, frame = cap.read()
    print(ret)
    if not ret:
        break
    frame = cv2.flip(frame, 0 ) #if video is coming upside down remove this

    imgS = cv2.resize(frame, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        print(1)
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(faceDist)
        if faceDist[matchIndex]<0.4:
            name = classNames[matchIndex].upper().lower()
            d[name]+=1
            if d[name]>5 and sent[name]==0:
                send_message(name)
                sent[name]=1
            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('webcam', frame)
    video.append(frame)
    height, width, layers = frame.shape
    size = (width,height)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out = cv2.VideoWriter('_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in range(len(video)):
# writing to a image array
    out.write(video[i])
out.release()