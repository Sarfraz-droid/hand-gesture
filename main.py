import streamlit as st
import cv2
import numpy as np
import classifier

st.title('Sign Recognition')

# picture = st.camera_input("Take a picture")

model = classifier.GestureDetection()
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# def createModal():
#     model = keras.models.load_model('./model.h5')
#     return model

# model = createModal()
camera = cv2.VideoCapture(cv2.CAP_V4L2)

while run:
    _, opencv_image = camera.read()
    
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    
    arr = np.asarray(opencv_image)
    
    img = model.createImage(arr)
    
    try:
    
        results = model.recognize_gesture(img)
        landmarks = model.get_landmarks(results)        
        annotated_image = model.annotate(opencv_image, landmarks)    
        cv2.putText(annotated_image, landmarks[0].category_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        FRAME_WINDOW.image(annotated_image, channels="RGB")
    except:
        FRAME_WINDOW.image(opencv_image, channels="RGB")

    
    
