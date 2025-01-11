import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants for body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Model dimensions
inWidth = 368
inHeight = 368

# Load pre-trained model
try:
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Human Pose Estimation with OpenCV")

st.text("Upload a clear image with all body parts visible for accurate detection.")

# File uploader
img_file_buffer = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Load the image
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    DEMO_IMAGE = 'stand.jpg'  # Default demo image
    image = np.array(Image.open(DEMO_IMAGE))

# Display the original image
st.subheader('Original Image')
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider
thres = st.slider('Threshold for detecting key points', min_value=0, value=20, max_value=100, step=5) / 100

# Pose estimation function
def poseDetector(frame, threshold=0.2):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Only first 19 parts (excluding background)

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(frame, points[idFrom], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[idTo], 5, (0, 0, 255), -1)

    return frame

# Run pose estimation
try:
    output = poseDetector(image, threshold=thres)
    st.subheader('Pose Estimated')
    st.image(output, caption="Pose Estimated", use_column_width=True)
except Exception as e:
    st.error(f"Error during pose estimation: {e}")


st.subheader('Positions Estimated')
st.image(
       output, caption=f"Positions Estimated", use_container_width=True)
    
st.markdown('''
            # 
             
            ''')

