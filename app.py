import cv2
import numpy as np
import streamlit as st
import os
import io
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import matplotlib.pyplot as plt
from PIL import Image

# Define paths
PATH_TO_MODEL = './detect.tflite'
PATH_TO_LABELS = './labelmap.txt'

# Define the tflite_detect_images function
def tflite_detect_images(image, modelpath, lblpath, min_conf=0.5, txt_only=False):
    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Convert the uploaded image to numpy array
    

    # Convert the uploaded image to a PIL Image
    uploaded_image = Image.open(image)

    # Convert the PIL Image to a NumPy array
    image = np.array(uploaded_image)

    # Check the data type of the image
    #st.write(image.dtype)

    # Preprocess the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    float_input = (input_details[0]['dtype'] == np.float32)
    if float_input:
        input_mean = 127.5
        input_std = 127.5
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform object detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    # Display or save the image with detections
    if txt_only == False:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        st.image(image, caption="Object Detection Result", use_column_width=True)
        #plt.figure(figsize=(12, 16))
        #plt.imshow(image)
        #plt.axis('off')
        #st.pyplot()
        
    
    return 
    
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model_path, lbl_path, min_conf=0.5):
        self.model_path = model_path
        self.lbl_path = lbl_path
        self.min_conf = min_conf
        self.initialize_model()

    def initialize_model(self):
        # Load the label map into memory
        with open(self.lbl_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Load the TensorFlow Lite model into memory
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def transform(self, frame):
        # Convert the frame to a PIL Image
        pil_image = Image.fromarray(frame)

        # Perform object detection
        tflite_detect_images(pil_image, self.interpreter, self.labels, self.input_details, self.output_details,
                            self.height, self.width, self.min_conf)

        # Convert the processed frame back to a NumPy array
        frame = np.array(pil_image)

        return frame

# Main Streamlit app
def main():
    st.title('Object Detection using Webcam')

    use_webcam = st.checkbox("Use Webcam")
    if use_webcam:
         st.title('Object Detection using Webcam')

    
        min_conf_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)
        transformer = VideoTransformer(PATH_TO_MODEL, PATH_TO_LABELS, min_conf_threshold)

        # Display the webcam input and process frames using the transformer
        webrtc_streamer(key="example", video_transformer_factory=lambda: transformer)
        
           
    else:
        st.title('Object Detection using Image Upload')

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            min_conf_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)

        #if st.button('Start Detection'):
            tflite_detect_images(uploaded_image, PATH_TO_MODEL, PATH_TO_LABELS, min_conf_threshold)
            # Do further processing with detections if needed

if __name__ == '__main__':
    main()
