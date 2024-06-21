import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Load YOLOv3 model
def load_yolo_model():
    weights_path = "C:/Users/hesham/Downloads/  .weights"
    config_path = "C:/Users/hesham/Downloads/yolov3.cfg"

    net = cv2.dnn.readNet(weights_path, config_path)

    classes = []
    with open("C:/Users/hesham/Downloads/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

# Function to perform object detection
def detect_objects(net, classes, output_layers, image):
    # Convert image to RGB
    if image.shape[2] == 4:  # Check if the image has 4 channels
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    Height, Width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects_detected = []
    for i in range(len(boxes)):
        if i in indexes:
            objects_detected.append(classes[class_ids[i]])
    return objects_detected

    
# Main function for Streamlit app
def main():
    st.title("Component Detection using YOLOv3")
    st.write("Upload an image and click 'Analyse Image' to detect components.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert('RGB')  # Ensure the image is in RGB format
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyse Image"):
            st.write("Analyzing...")
            net, classes, output_layers = load_yolo_model()
            components = detect_objects(net, classes, output_layers, image)
            
            if components:
                st.subheader("Detected Components:")
                for component in components:
                    st.write(f"- {component}")
            else:
                st.write("No components detected.")

if __name__ == "__main__":
    main()