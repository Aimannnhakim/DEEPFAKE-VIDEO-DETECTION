import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from matplotlib.animation import FuncAnimation
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from tensorflow.keras.models import *

import sys


#matplotlib
#fig = plt.figure()





def calresult(x):
    if x==0 :
        return 1;
    else :
        return 0;


def printresult(x):

    if x>50 :
        return "DEEPFAKE"
    else :
        return "REAL"



def detect(dirr):

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    input_shape = (128, 128, 3)
    pr_data = []
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(dirr)
    frameRate = cap.get(5)

    calc=0.0;
    count=0;


    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()

        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                
                try:
                    imgresize = cv2.resize(crop_img, (128,128))
                except cv2.error as e:
                    continue
             
                data=img_to_array(imgresize).flatten() / 255.0
                
                #imgplot = plt.imshow(crop_img)
                #plt.show()
                #plt.close()
                
                data = np.array(data)
                count+=1;
                #print ("Count",count,"calc",calc)
              
                
                data = data.reshape(-1, 128, 128, 3)

                calc+=calresult(loaded_model.predict_classes(data))

                

                                
    val= calc/count *100     
    return printresult(val)
                



#print(printresult(detect('D:/deepfake-detection-masterr/deepfake-detection-master/deepfake-detection-challenge/test_videos/ggdpclfcgk.mp4')))
                
     