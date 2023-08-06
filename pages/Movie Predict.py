import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/leaf.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a file")


map_dict = {0:'Canker',
            1:'Dot',
            2:'Mummification',
            3:'Rust'
            }


if uploaded_file is not None:
    # Convert the file to an Rust image.
    # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # opencv_image = cv2.imdecode(file_bytes, 1)
    # opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.video(uploaded_file)



    # resized = mobilenet_v2_preprocess_input(resized)
    # img_reshape = resized[np.newaxis,...]
    video_pred = st.button("Set Video")  
    if video_pred:
       myFrameNumber = 50
       cap = cv2.VideoCapture("video.mp4")

    # get total number of frames
       totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # check for valid frame number
       if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
    # set frame position
       cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)

       while True:
          ret, frame = cap.read()
          cv2.imshow("Video", frame)
          if cv2.waitKey(20) & 0xFF == ord('q'):
             break

       cv2.destroyAllWindows()

    Genrate_pred = st.button("Leaf Predict")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Dryness Level for the image is {}".format(map_dict [prediction]))
 
           
