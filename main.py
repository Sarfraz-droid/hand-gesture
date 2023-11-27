import streamlit as st
import cv2
import numpy as np
import classifier
from streamlit_webrtc import webrtc_streamer
import av

st.title('Sign Recognition')

# picture = st.camera_input("Take a picture")

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
        cv2.putText(annotated_image, landmarks[0].category_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")
    except:
        return av.VideoFrame.from_ndarray(opencv_image, format="rgb24")
        
webrtc_streamer(key="sign_detection", video_frame_callback=video_frame_callback)
