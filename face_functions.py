#!/usr/bin/env python
# coding: utf-8

# # Instructions to use face_functions.py
# 
# ## This python file contains several functions which can be called by:
# 
#     1. import face_functions.py
#     2. Use any function in here by calling face_functions.function-name(args) at another IDE.
#    
# ## List of Functions Available:
# 
#     1. get_faces_labels(image_path): return arrays of 100x100 grayscale faces, labels, and dictionary {name:label}
#     2. face_recognition(): real time face recognition without mask.
#     3. face_mask_recognition(): real time face recognition with mask.
#     4. to_rgb(image): return grayscale image.
#     5. face_detection(cascade, color_img, scaleFactor): return grayscale image and color face roi. 
#     6. face_eyes_detection(cascade1, cascade2, color_img, scaleFactor): return grayscale image and color face roi. 
#     7. face_eyes_smile_detection(cascade1, cascade2, cascade3, color_img, scaleFactor): return same as above.

# In[3]:


import os, sys
import cv2
from PIL import Image
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Get Face Images and Face Labels from Resized_Faces for Face Training
def get_faces_labels(resized_images_path='Resized_Faces'):
    file_path = os.listdir(resized_images_path)
    faces = []
    face_labels = []
    current_id=0
    label_ids={}
    for file in file_path:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(resized_images_path, file)
            label=os.path.basename(path).split(".")[1]
            #print(label,path)
            if not label in label_ids:
                #label_ids[label] = os.path.basename(path).split(".")[2]
                label_ids[label]=current_id
                current_id+=1

            id_=label_ids[label]
            pil_image=Image.open(path)#grayscale
            image_array=np.array(pil_image,'uint8')
                           
            #print(image_array)
            faces.append(image_array)
            face_labels.append(id_)
            faces_array = np.array(faces)
            labels_array = np.array(face_labels)
    return faces_array, labels_array, label_ids


# In[6]:


# Real-Time Detection using webcam after training the face recognizer
def face_recognition():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load classifier
    eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eye_glassesCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    #faceCascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml') # load classifier

    cap = cv2.VideoCapture(0)
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("LBPH_Face_Recognizer.yml")

    #Load face labels
    # with open('face_eye_smile_labels.pickle', 'rb') as f:
    #     y_train = np.array(pickle.load(f))
    names = ['Bryan_Lee', 'Bryan_Lim', 'Edmund', 'Malvern', 'Ter_Ren', 
             'Wang_Jue', 'Yi_Cheng', 'Yi_Rong']
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(5, 5)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
    #        roi_gray = cv2.resize(roi_gray, (100,100))
            eyes = eye_glassesCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5, minSize=(5,5))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    #         smile = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=25, minSize=(120,120))
    #         for (xx, yy, ww, hh) in smile:
    #             cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
            gray_face = cv2.resize((gray[y:y+h,x:x+w]),(100,100))
            label, conf = face_recognizer.predict(gray_face)

            if conf<=110:
                person = names[label]
            else:
                person = "Unknown"

            text = str(label) + person + ":" + str(round(conf,3))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('frame',frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


# In[1]:


# With mask on
# Real-Time Detection using webcam after training the face recognizer
def face_mask_recognition():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load classifier
    eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eye_glassesCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    #faceCascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml') # load classifier

    cap = cv2.VideoCapture(0)
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("LBPH_Face_Recognizer.yml")

    #Load face labels
    # with open('face_eye_smile_labels.pickle', 'rb') as f:
    #     y_train = np.array(pickle.load(f))
    names = ['Bryan_Lee', 'Bryan_Lim', 'Edmund', 'Malvern', 'Ter_Ren', 
             'Wang_Jue', 'Yi_Cheng', 'Yi_Rong']
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyes = eye_glassesCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(5,5))
        for (ex, ey, ew, eh) in eyes:
            roi_gray = gray[ey:ey+eh, ex:ex+ew]
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    #         smile = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=25, minSize=(120,120))
    #         for (xx, yy, ww, hh) in smile:
    #             cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
        #gray_eyes = cv2.resize((gray[y:y+h,x:x+w]),(100,100))
        label, conf = face_recognizer.predict(gray)

        if conf<=145:
            person = names[label]

        else:
            person = "Unknown"

        text = str(label) + person + ":" + str(round(conf,3))
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        cv2.putText(frame, text, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('frame',frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


# # Face Detection

# In[4]:


# To show coloured image using matplotlib, image must be converted to RGB. Default is BGR.
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[31]:


def face_detection(cc, color_img, scaleFactor=1.2):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = color_img.copy()
    #convert the color image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
    #let's detect all faces in the image (some faces may be closer to camera than others) images
    faces = cc.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=5, minSize=(20,20));   
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img_copy[y:y+h, x:x+w]

    return gray_img, faces


# In[32]:


def face_eyes_detection(cc1, cc2, color_img, scaleFactor=1.2):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = color_img.copy()
    #convert the color image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
    #let's detect all faces in the image (some faces may be closer to camera than others) images
    faces = cc1.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=5);   
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img_copy[y:y+h, x:x+w]
        eyes = cc2.detectMultiScale(roi_gray, scaleFactor=scaleFactor, minNeighbors=5);
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 5)
    return gray_img, faces


# In[2]:


def face_eyes_smile_detection(cc1, cc2, cc3, color_img, scaleFactor=1.2):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = color_img.copy()
    #convert the color image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)       
    #let's detect all faces in the image (some faces may be closer to camera than others) images
    faces = cc1.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=5);   
    #go over list of faces and draw them as rectangles on original colored img

    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img_copy[y:y+h, x:x+w]
        eyes = cc2.detectMultiScale(roi_gray, scaleFactor=scaleFactor, minNeighbors=20, minSize=(50,50));
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smile = cc3.detectMultiScale(roi_gray, scaleFactor=scaleFactor, minNeighbors=25, minSize=(120,120));
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_gray, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
        #resized_img = cv2.resize(gray_img[y:y+h, x:x+w], (200,200))
    return gray_img, faces


# In[5]:


def process_image(image_folder_path="Faces", size=(100,100)):
    dirs = os.listdir(image_folder_path)
    miss_faces = []
    faces = []
    labels = []
    count = 0
    
    for dir_name in dirs:
        if not dir_name.startswith("face"):
            continue;
        label = int(dir_name.replace("face",""))
        subject_dir_path = image_folder_path + "/" + dir_name
        # get the folder name that contains each person eg Faces/face1
        subject_images_names = os.listdir(subject_dir_path)
        # go through each image in person folder
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            count+=1
            print(count)
            #resize_image = cv2.resize(image,size)
            # display an image window to show the image
            #cv2.imshow("Training on image...", resize_image)
            cv2.waitKey(100)
            # Face Detection
            #face, rect_face = face_detection(haar_face_cascade, image)
            # Face and Eyes Detection
            #face, rect_face = face_eyes_detection(haar_face_cascade, haar_eye_cascade, image)
            # Face, Eyes, Smile Detection
            face, rect_face = face_eyes_smile_detection(haar_face_cascade, haar_eye_cascade, haar_smile_cascade, image)
            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                miss_faces.append(face)
    with open('face_eye_smile_labels.pickle', 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(faces, f)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return miss_faces, faces, labels

