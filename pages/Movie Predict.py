import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/movie.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a file")
st.write(uploaded_file)
frame_skip = 300

map_dict = {0:'Action',
            1:'Comedy',
            2:'Fantacy',
            3:'Horror'
            }


if uploaded_file is not None:
    # Convert the file to an Rust image.
    st.write(uploaded_file.type)
    if uploaded_file.type == 'image/png':
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      opencv_image = cv2.imdecode(file_bytes, 1)
      opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
      resized = cv2.resize(opencv_image,(224,224))
      st.image(opencv_image, channels="RGB")
      resized = mobilenet_v2_preprocess_input(resized)
      img_reshape = resized[np.newaxis,...]
    else:
    # Now do something with the image! For example, let's display it:
      st.video(uploaded_file)



    video_pred = st.button("Set Video")  
    if video_pred:
       vid = uploaded_file.name
       with open(vid, mode='wb') as f:
         f.write(uploaded_file.read()) # save video to disk

       st.markdown(f"""
         ### Files
         - {vid}
         """,
       unsafe_allow_html=True) # display file name

       vidcap = cv2.VideoCapture(vid) # load video from disk
       cur_frame = 0
       success = True

       while success:
           success, frame = vidcap.read() # get next frame from video
           if cur_frame % frame_skip == 0: # only analyze every n=300 frames
              print('frame: {}'.format(cur_frame)) 
              pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
              st.image(pil_img)
              cur_frame += 1
              # file_bytes = np.asarray(frame, dtype=np.uint8)
              # opencv_image = cv2.imread(pil_img,1)
              # opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
              # resized = cv2.resize(opencv_image,(224,224))
              # st.image(opencv_image, channels="RGB")
              resized = mobilenet_v2_preprocess_input(pil_img)
              img_reshape = resized[np.newaxis,...]


    Genrate_pred = st.button("Genere Predict")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.session_state['year'] = prediction
        st.session_state['genre'] = 'Action'
        st.write(st.session_state.year)
        st.title("Predicted Movie genere is {}".format(map_dict [prediction]))
 
           
