#!/usr/bin/env python
# coding: utf-8

# # Instructions to use face_functions.py
# 
# ## This python file contains several functions which can be called by:
# 
#     1. Download this notebook as python file (.py).
#     2. import face_functions at another IDE.
#     3. Use any function in here by calling face_functions.function-name(args) at another IDE.
#    
# ## List of Functions Available:
# 
#     1. get_faces_labels(image_path): return arrays of 100x100 grayscale faces, labels, and dictionary {name:label}
#     2. get_faces_labels_pca(image_path): return arrays of 1D grayscale faces, labels, and dictionary {name:label}
#     3. normalise_face(face_array, n, m): return average face (1,10000), zeromean face (n,10000), eigenvalues (1,m), eigenvectors (10000,m), eigenfaces (m,10000), pca faces (n,m), covariance matrix (10000,10000)
#     4. normalise_test_face(test_face_array, avg_face, eigenvectors): return zeromean test face(1,10000), pca test face (1,m). 
#     5. plot_gallery(): return plots of eigenfaces. 
#     6. face_recognition(): real time face recognition without mask via LBPH Face Recognizer.
#     7. face_mask_recognition(): real time face recognition with mask via LBPH Face Recognizer.
#     8. pca_face_recognition(): real time pca face recognition without mask using SVM.
#     9. to_rgb(image): convert bgr to rgb.
#     10. face_detection(cascade, color_img, scaleFactor): return grayscale image and color face roi. 
#     11. face_eyes_detection(cascade1, cascade2, color_img, scaleFactor): return grayscale image and color face roi. 
#     12. face_eyes_smile_detection(cascade1, cascade2, cascade3, color_img, scaleFactor): return same as above.
#     13. illumination_normalize(rgb_image_array): return rgb face array (n,100,100), ycrcb face array (n, 100, 100).
#     14. dimension_reduction(face_encoding, n, m): return eigenvalues, eigenvector (2622,m), face_train (n,m), covariance matrix (2622,2622).
#     15. findCosineSimilarity(face_database, test_face): return scalar value ranges between 0 and 1.
#     16. vgg_face_recognition(name_array, face_train, eigenvector): real time face recogniton using Euclidean Distance.
#     17. lbph_face_recognition(): real time face recognition with or without mask.
# 
# **Note: #3 In normalise_face(), change the n & m values according for different number of samples and top m features, respectively.**
# 
# **Note: In #14 dimension_reduction(), change the n & m values according for different number of samples and top m features, respectively.**
# 
# **Note: In #5 plot_gallery(), Change the n_row and n_col accordingly. n_row * n_col should be equal to the m value.**

# In[2]:


# Import dependencies
import os, sys
import cv2
from PIL import Image
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#1 Get Face Images and Face Labels from Resized_Faces for Face Training
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


# In[3]:


#2 Get Face Images and Face Labels from Resized_Faces for PCA Training
def get_faces_labels_pca(resized_images_path='Resized_Faces'):
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
            #reshape 2D array into 1D array
            image_1d = image_array.reshape(-1)
                           
            #print(image_array)
            faces.append(image_1d)
            face_labels.append(id_)
            faces_array = np.array(faces)
            labels_array = np.array(face_labels)
    return faces_array, labels_array, label_ids


# In[1]:


#3 Process training faces for training SVM
def normalise_face(image, n=100, m=40):
    # Find average face based on face dataset, return shape (100, 10000)
    avg_face = np.mean(image, axis=0)
    print("avg face (1,10000): ", avg_face.shape)
    # Compute zero mean faces, return shape (100, 10000)
    zeromean_face = image - avg_face
    print("zero mean face (n,10000): ", zeromean_face.shape)
    # Compute covariance matrix, return shape (10000, 10000)
    covariance = np.dot(zeromean_face.T, zeromean_face) / n
    print("covariance matrix shape (10000, 10000): ", covariance.shape)
    total_features = image.shape[1] # 10000 features
    print("Calculaing top eigenvalues and corresponding eigenvectors...")
    # Compute eigenval and eigenvec, return shape (1, m) and (10000, m)
    eigenvalues, eigenvectors = linalg.eigh(covariance, eigvals=(total_features-m,total_features-1))
    print("eigenvalues shape (m,): ", eigenvalues.shape)
    print("eigenvectors shape (10000, m): ", eigenvectors.shape)
    print("Eigens Computation Done!")
    # Compute eigenfaces, return shape (m, 10000)
    eigenfaces = eigenvectors.T
    print("eigenface shape (m,10000)", eigenfaces.shape)
    # Project zero mean faces into eigen space for training, return shape (100, m)
    face_train = np.dot(zeromean_face, eigenvectors)
    print("face_train (n,m): ", face_train.shape)
    return avg_face, zeromean_face, eigenvalues, eigenvectors, eigenfaces, face_train, covariance


# In[4]:


#4 Process real time detection face (per detected face)
def normalise_test_face(image, avg_face, eigenvectors):
    zeromean_testface = image - avg_face
    face_test = np.dot(zeromean_testface, eigenvectors)
    return zeromean_testface, face_test


# In[10]:


#5 Plot eigenfaces
def plot_gallery(images, titles, h, w, n_row=8, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())


# In[2]:


#6 Real-Time Detection using webcam after training the face recognizer
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
    names = ['Bryan_Lee', 'Bryan_Lim', 'Chin_Fung', 'Darren', 'Edmund', 'John', 'Lam', 
             'Malvern', 'meinv', 'Nicholas', 'Peter', 'Ter_Ren', 'Wang_Jue', 'Yi_Cheng', 
             'Yi_Rong', 'Yong_Zhe', 'Yuan_Jun', 'Zhi_jia', 'Zi_Hang', 'Zi_Ying']
    
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


# In[3]:


# With mask on
#7 Real-Time Detection using webcam after training the face recognizer
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
    names = ['Bryan_Lee', 'Bryan_Lim', 'Chin_Fung', 'Darren', 'Edmund', 'John', 'Lam', 
             'Malvern', 'meinv', 'Nicholas', 'Peter', 'Ter_Ren', 'Wang_Jue', 'Yi_Cheng', 
             'Yi_Rong', 'Yong_Zhe', 'Yuan_Jun', 'Zhi_jia', 'Zi_Hang', 'Zi_Ying']
    
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


# In[4]:


#8 Real Time Detection using SVM
def pca_face_recognition(avgface, eigenvectors):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load classifier
    eye_glassesCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    
    # Load trained model
    with open('svc_rbf_10_pca.pickle', 'rb') as saved_model:
        svc = pickle.load(saved_model)

    cap = cv2.VideoCapture(0)

    names = ['Bryan_Lee', 'Bryan_Lim', 'Chin_Fung', 'Darren', 'Edmund', 'John', 'Lam', 
             'Malvern', 'meinv', 'Nicholas', 'Peter', 'Ter_Ren', 'Wang_Jue', 'Yi_Cheng', 
             'Yi_Rong', 'Yong_Zhe', 'Yuan_Jun', 'Zhi_jia', 'Zi_Hang', 'Zi_Ying']
    
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
            gray_face_1d = gray_face.reshape(1,-1)
            #print(gray_face_1d.shape)
            # Process test face the same as trained faces
            #train_face, _, _ = get_faces_labels_pca()
            #avgface, _, _, eigenface = normalise_face1(train_face)
            normface_test, test_face = normalise_test_face1(gray_face_1d, avgface, eigenvectors)
            #plt.imshow(normface_test.reshape(100,100), cmap='gray')
            y_pred = svc.predict_proba(test_face)
            #top_prob = max(y_pred)
            top_prob_name = names[np.argmax(y_pred)]
            top_prob = y_pred[:,np.argmax(y_pred)]
            print(y_pred)
            #print(top_prob)
            
            #text = str(y_pred[0]) + ':' + names[int(y_pred[0])]
            text = str(top_prob_name) + "-Prob: " + str(top_prob)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


# # Face Detection

# In[13]:


#9 To show coloured image using matplotlib, image must be converted to RGB. Default is BGR.
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[14]:


#10
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


# In[15]:


#11
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


# In[16]:


#12
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


# In[1]:


#13
def illumination_normalize(rgb_image_array):
    image_rgb=[]
    image_ycrcb=[]
    
    for i in rgb_image_array:
        image_reshape = cv2.resize(i, dsize=(100,100), interpolation=cv2.INTER_AREA)
        ycrcb_image = cv2.cvtColor(image_reshape, cv2.COLOR_RGB2YCrCb) # Convert rgb image to ycrcb image
        # separate channels
        y, cr, cb = cv2.split(ycrcb_image)

        # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
        # account for size of input vs 300
        sigma = int(5 * 100 / 100)
        #print('sigma: ',sigma)
        gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

        # subtract background from Y channel
        y = (y - gaussian + 100)

        # merge channels back
        ycrcb = cv2.merge([y, cr, cb])

        #convert to RGB
        output_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        image_rgb.append(output_rgb)
        image_rgb_array = np.array(image_rgb)
        image_ycrcb.append(ycrcb_image)
        image_ycrcb_array = np.array(image_ycrcb)

    return image_rgb_array, image_ycrcb_array


# In[5]:


#14
def dimension_reduction(face_encoding, n=100, m=100):
    # Compute covariance matrix, return shape (2622, 2622)
    covariance = np.dot(face_encoding.T, face_encoding) / n
    print("covariance matrix shape (2622, 2622): ", covariance.shape)
    total_features = face_encoding.shape[1] # 2622 features
    print("Calculaing top eigenvalues and corresponding eigenvectors...")
    # Compute eigenval and eigenvec, return shape (1, m) and (2622, m)
    eigenvalues, eigenvectors = linalg.eigh(covariance, eigvals=(total_features-m,total_features-1))
    print("eigenvalues shape (m,): ", eigenvalues.shape)
    print("eigenvectors shape (2622, m): ", eigenvectors.shape)
    print("Eigens Computation Done!")
    # Compute eigenfaces, return shape (m, 2622)
    eigenfaces = eigenvectors.T
    print("eigenface shape (m,2622)", eigenfaces.shape)
    # Project zero mean faces into eigen space for training, return shape (100, m)
    face_train = np.dot(face_encoding, eigenvectors)
    print("face_train (100,m): ", face_train.shape)
    return eigenvalues, eigenvectors, eigenfaces, face_train, covariance


# In[7]:


#15
def findCosineSimilarity(face_database, test_face):
    a = np.matmul(np.transpose(face_database), test_face)
    b = np.sum(np.multiply(face_database, face_database))
    c = np.sum(np.multiply(test_face, test_face))
    return (a / (np.sqrt(b) * np.sqrt(c)))


# In[8]:


#16
def vgg_face_recognition(name_array, face_train, eigenvector):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load face classifier
    eye_glassesCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml') # Load eye classifier
    
    with open('vggface_model.pickle', 'rb') as vggface:
        custom_vgg_model = pickle.load(vggface)
    
    cap = cv2.VideoCapture(0)

    cap.set(3,1280) # set Width
    cap.set(4,960) # set Height
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
            resized_color = cv2.resize(frame[y:y+h,x:x+w], dsize=(224,224), interpolation=cv2.INTER_AREA)
            eyes = eye_glassesCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5, minSize=(5,5))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    #         smile = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=25, minSize=(120,120))
    #         for (xx, yy, ww, hh) in smile:
    #             cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
            rgb_face_resize = np.reshape(resized_color, (1,224,224,3))
            y_pred = custom_vgg_model.predict(rgb_face_resize) # 1x2622
            captured_representation = np.dot(y_pred, eigenvector) # 1x100
            
            
            found=0
            max_value=0
            for i in range(len(name_array)):
                similarity = findCosineSimilarity(face_train[i], captured_representation.T)
                if((similarity > 0.8) & (similarity > max_value)):
                    max_value = similarity
                    name = name_array[i]
                    found=1
            if (found==1):
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print(name, max_value)
             
            elif (found==0):
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print('Unknown', similarity)
                
        cv2.imshow('frame',frame)
        k = cv2.waitKey(10) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


# In[1]:


#17 Real-Time Detection using webcam after training the face recognizer
def lbph_face_recognition():
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
    names = ['Bryan_Lee', 'Bryan_Lim', 'Chin_Fung', 'Darren', 'Edmund', 'John', 'Lam', 
             'Malvern', 'meinv', 'Nicholas', 'Peter', 'Ter_Ren', 'Wang_Jue', 'Yi_Cheng', 
             'Yi_Rong', 'Yong_Zhe', 'Yuan_Jun', 'Zhi_jia', 'Zi_Hang', 'Zi_Ying']
    
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
        found=0
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
            
            found=1
            
        if (found==1):

            if conf<=120:
                person = names[label]
            else:
                person = "Unknown"

            text = str(label) + person + ":" + str(round(conf,3))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        elif (found==0):
            eyes = eye_glassesCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(5,5))
            for (ex, ey, ew, eh) in eyes:
                roi_gray = gray[ey:ey+eh, ex:ex+ew]
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        #         smile = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=25, minSize=(120,120))
        #         for (xx, yy, ww, hh) in smile:
        #             cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
            #gray_eyes = cv2.resize((gray[y:y+h,x:x+w]),(100,100))
                label, conf = face_recognizer.predict(gray)
            if conf<=120:
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


# In[ ]:




