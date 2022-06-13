import cv2
import os
import Augmentor

def preprocess_img(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    a,b=gray.shape
    
    new_a=a//2
    new_b=b//2
    
    resized=cv2.resize(gray,(new_b,new_a))
    return resized

def detect_face(img,face_cascade):
    
    faces= face_cascade.detectMultiScale(img,1.1,4)
    if len(faces)==0:
        return None,False
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    a,b=img.shape
    crop_img = img[y:y+h, x:x+w]
#     cv2.imshow("cropped",crop_img)
    crop_img=cv2.resize(crop_img,(b*2,a*2))
    
    return crop_img,True

def crop_images(source_folder,destination_folder):
    for i in os.listdir(source_folder): 
        if i not in os.listdir(destination_folder):
            os.mkdir(destination_folder + "/"+i)

        for j in os.listdir(source_folder + "/"+ i):
            img_path = source_folder+'/'+ i+'/'+j

            image=cv2.imread(img_path)
#             cv2.imshow("s",image)
            resized2=preprocess_img(image)

            detected_img,is_detected=detect_face(resized2,face_cascade)
            if is_detected:
                cv2.imwrite(destination_folder+'/'+i+'/'+j, detected_img)

def augment(source_folder):
    for i in os.listdir(source_folder):
        if len(os.listdir(source_folder + "/"+i))==0:
            continue 
        p = Augmentor.Pipeline(source_folder + '/'+ i, output_directory=source_folder + "/"+i)

        # Defining augmentation parameters and generating 5 samples
        p.flip_left_right(0.5)
        # p.black_and_white(0.1)
        p.rotate(0.3, 10, 10)
        p.skew(0.4, 0.5)
        p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5)
        p.sample(100-len(os.listdir(source_folder + "/"+i)))

face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

base_path="C:/facedetect/"
source_folder =base_path+'temp'
destination_folder = base_path+'final'

crop_images(source_folder,destination_folder)
augment(destination_folder)