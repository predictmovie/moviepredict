import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/leaf.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a file")


map_dict = {0:'Nun',
            1:'Metrix',
            2:'Avengers',
            3:'Rust'
            }


if uploaded_file is not None:
    # Convert the file to an Rust image.
    st.write(uploaded_file.type)
    if uploaded_file.type == 'image/png':
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    else:
    # Now do something with the image! For example, let's display it:
    st.video(uploaded_file)



    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]
    video_pred = st.button("Set Video")  
    if video_pred:
        cap = cv2.VideoCapture(video_name)

#Set frame_no in range 0.0-1.0
#In this example we have a video of 30 seconds having 25 frames per seconds, thus we have 750 frames.
#The examined frame must get a value from 0 to 749.
#For more info about the video flags see here: https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
#Here we select the last frame as frame sequence=749. In case you want to select other frame change value 749.
#BE CAREFUL! Each video has different time length and frame rate. 
#So make sure that you have the right parameters for the right video!
        time_length = 30.0
        fps=25
        frame_seq = 749
        frame_no = (frame_seq /(time_length*fps))

#The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
#Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
#The second argument defines the frame number in range 0.0-1.0
        cap.set(2,frame_no);

#Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
        ret, frame = cap.read()

#Set grayscale colorspace for the frame. 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Cut the video extension to have the name of the video
        my_video_name = video_name.split(".")[0]

#Display the resulting frame
        cv2.imshow(my_video_name+' frame '+ str(frame_seq),gray)

#Set waitKey 
        cv2.waitKey()

#Store this frame to an image
        cv2.imwrite(my_video_name+'_frame_'+str(frame_seq)+'.jpg',gray)

# When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


    Genrate_pred = st.button("Genere Predict")    
    if Genrate_pred:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image,(224,224))
        prediction = model.predict(img_reshape).argmax()
        st.session_state['year'] = prediction
        st.session_state['genre'] = 'Action'
        st.write(st.session_state.year)
        st.title("Predicted Movie genere is {}".format(map_dict [prediction]))
 
           
