import keras
from skimage import transform
import numpy as np
import making_dataset
import cv2 
from message import send_message

face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
d={'anuhya':0,'madhu':0,'pavitra':0,'prerana':0}

def get_image(img):
    resized = making_dataset.preprocess_img(img)
    cropped,is_face= making_dataset.detect_face(resized,face_cascade)
    np_image = np.array(cropped).astype('float32')/255
    np_image = transform.resize(np_image, (64, 64, 1))
    np_image = np.expand_dims(np_image, axis=0) #?
    return np_image

model = keras.models.load_model('model.h5')

cap = cv2.VideoCapture('VID-20220607-WA0012.mp4') 
count = 0
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)

font = cv2.FONT_HERSHEY_SIMPLEX
# Line thickness of 2 px
thickness = 2
while True:
    count+=1
    if count%20 !=0:####################################### why
        continue
    # Capture frame-by-frame'
    ret, frame = cap.read()#ret- boolean if any return,cap.read infinite loop

    if not ret:
        break
    np_images=get_image(frame)
    print(list(model.predict(np_images)[0]))
    res = list(model.predict(np_images)[0])
    print(max(res))
    face = res.index(max(res))
    print(face)
    if res[face]>0.9:
        if face ==0:
            frame = cv2.putText(frame, 'Anuhya', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            d['anuhya']+=1
            if d['anuhya'] >=5:
                send_message('anuhye')
                d['anuhya'] =0
        elif face ==1: 
            frame = cv2.putText(frame, 'Madhu', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            d['madhu']+=1
            if d['madhu'] >=5:
                send_message('madhu')
                d['madhu'] =0
        elif face ==2:
            frame = cv2.putText(frame, 'Pavitra', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            d['pavitra']+=1
            if d['pavitra'] >=5:
                send_message('pavitra')
                d['pavitra'] =0
        elif face ==3:
            frame = cv2.putText(frame, 'Prerana', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            d['prerana']+=1
            if d['prerana'] >=5:
                send_message('prerana')
                d['prerana']=0
#     resized = preprocess_img(frame)
#     cropped,is_face=detect_face(resized,face_cascade)    
#     cropped = get_image(cropped)
#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
