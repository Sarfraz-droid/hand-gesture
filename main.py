import streamlit as st
import cv2
import numpy as np
import classifier
from streamlit_webrtc import webrtc_streamer
import av
from dotenv import load_dotenv
import threading
from get_gpt_prediction import get_gpt_prediction
import random


load_dotenv()

st.title('Sign Recognition')

# picture = st.camera_input("Take a picture")

prediction = []

model = classifier.GestureDetection()

def video_frame_callback(frame):
    
    opencv_image = frame.to_ndarray(format="bgr24")
    
    
    if(opencv_image is None):
        return av.VideoFrame.from_ndarray(opencv_image, format="bgr24")
    
    opencv_image = cv2.flip(opencv_image, 1)
    
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    img = model.createImage(opencv_image)
    
    try:
    
        results = model.recognize_gesture(img)
        landmarks = model.get_landmarks(results)        
        annotated_image = model.annotate(opencv_image, landmarks)   
        category = landmarks[0].category_name
        
        # print(category)
        
        if category != 'None':            
            if(len(prediction) > 0):
                if(prediction[-1] != category):
                    prediction.append(category)
                
                if(len(prediction) > 4):
                    prediction.pop(0)
            else:
                prediction.append(category)
            
        # print(prediction)
        cv2.putText(annotated_image, landmarks[0].category_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")
    except:
        return av.VideoFrame.from_ndarray(opencv_image, format="rgb24")
        
webrtc_streamer(key="sign_detection", video_frame_callback=video_frame_callback)


def setInterval(func,time):
    e = threading.Event()
    while not e.wait(time):
        func()
        
text_area = st.empty()
        
def update_prediction():
    global prediction
    print(prediction)
    try:
        if(len(prediction) > 0):
            text = get_gpt_prediction(' '.join(prediction))
            text_area.text_area("Prediction", text, key=random.randint(0, 10000000))
        else:
            text_area.text_area("Prediction", "No prediction yet", key=random.randint(0, 10000000))
    except:
        text_area.text_area("Prediction", "No prediction yet", key=random.randint(0, 10000000))
        
setInterval(update_prediction, 1)
