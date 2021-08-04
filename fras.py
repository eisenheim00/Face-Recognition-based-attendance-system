

import pickle
import pandas as pd
from datetime import date
from csv import writer
import cv2
import csv
import os
from PIL import Image
import numpy as np

faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
images_path = "project_dataset"
trainX = []
trainY = []
label_id = 0
labelIds = {}
for root,path,file in os.walk(images_path): #search in dataset
    for f in file:
        if f.endswith('png') or f.endswith('.jpg') or f.endswith('JPG')or f.endswith('jpeg'):
            path = os.path.join(root,f)
            label = os.path.basename(os.path.dirname(path)).replace(" ",'.').lower() #add labels
            #print(label,path)
            
            if not label in labelIds: #changed 
                labelIds[label] = label_id
                label_id = label_id + 1
                
                    
            id_ = labelIds[label]
            #print(labelIds)
            pillowImg = Image.open(path).convert("L") #greyscale conversion
            imgArr = np.array(pillowImg,"uint8") #converting immg to np array (into numbers) used for training
            #print(imgArr)
            #size reset
            size = (128,128)
            testingImg = pillowImg.resize(size,Image.ANTIALIAS)
            imgArr = np.array(testingImg,'uint8')
            #ROI in training data
            
            faces = faceClassifier.detectMultiScale(imgArr, scaleFactor = 1.6, minNeighbors = 6)
            for (i,j,width,height) in faces:
                regionOfInterest = imgArr[j:j+height,i:i+width]
                trainX.append(regionOfInterest)
                trainY.append(id_)


#save the labels to use in main

with open('Ydata.pickle' , 'wb') as f:
    pickle.dump(labelIds,f) #bytstream conversion to store

#training the model
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainY = np.array(trainY)
recognizer.train(trainX,trainY)
recognizer.save('trainer.yml')

#implementing the recognizer and using it in above portion

#load names from pickle in original work

import csv
def append_list_as_row(fname, list_of_elem):
    # Open file in append mode
    with open(fname, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#att_reg = pd.read_csv('att_reg.csv')

#read csv, and split on "," the line





recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
lbs={"person":1}
#load training data
with open("Ydata.pickle",'rb') as f:
    original = pickle.load(f)
    lbs = {v:k for k,v in original.items()} #invert

vid = cv2.VideoCapture(0)
count = 0
while(True):
    rate, frames = vid.read() #frame by frame reading
    grayImage = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grayImage, scaleFactor = 1.6, minNeighbors =6)
    for (i,j,width,height) in faces:
        regionOfInterest_clr = frames[j:j+height,i:i+width]
        regionOfInterest_gray = grayImage[j:j+height,i:i+width]#take fave only i.e. region of interest
        #predict
        id_, conf = recognizer.predict(regionOfInterest_gray)
        
        #putting text
        if conf > 80:
            #print(id_)
            #print(lbs[id_])
            cv2.putText(frames,lbs[id_],(i,j),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            count = count+1
            today = date.today()
            # dd/mm/YY
            d1 = today.strftime("%d/%m/%Y")
            rowData = [lbs[id_],d1]
            #loop through the csv list
            isPresent = False
            if count == 60:
                csv_file = csv.reader(open('att_reg.csv', "r"), delimiter=",")
                for row in csv_file:
                #if current rows 2nd value is equal to input, print that row
                    if d1 == row[1] and lbs[id_] == row[0] :
                        isPresent = True
                        break
                if isPresent == False:
                    append_list_as_row('att_reg.csv', rowData)
                count = 0
        img = '8.jpg'
        cv2.imwrite(img,regionOfInterest_clr)
        thick = 1 #thickness
        clr = (100,100,255) #color

        cv2.rectangle(frames,(i,j),(i+width,height+j),clr,thick)
        eyes = eye.detectMultiScale(regionOfInterest_gray)
        for (x,y,w,h) in eyes:
            cv2.rectangle(regionOfInterest_clr,(x,y),(x+w,h+y),(100,255,100),1)
    cv2.imshow('Attendance System',frames) #display
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows





