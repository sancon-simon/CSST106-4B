# 4B-SANCON-EXER4
## Sancon, Simon B - BSCS4B

## Important Links
1. [Actual Notebook](https://colab.research.google.com/drive/1S7EYDoqql4NAsAsFb1_713-h5H42PkoV?usp=sharing)
2. [Repository Notebook](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-EXER4/code/4B_SANCON_EXER4.ipynb)
3. [Image Folder](https://github.com/sancon-simon/CSST106-4B/tree/main/4B-SANCON-EXER4/images)
4. [Performance Analysis](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-EXER4/performance_analysis/4B-SANCON-EXER4-performance_analysis.md)

## Hands On Exploration:

In this Exercise the following feature extraction and Object detection was showcased

1. HOG (Histogram of Oriented Gradients) Object Detection
  * HOG is a feature descriptor widely used for object detection, particularly for human detection.
  * HOG focuses on the structure of objects through gradients.
  * Useful for detecting humans and general object recognition
      
2. YOLO (You Only Look Once) Object Detection
  * YOLO is a deep learning-based object detection method.
  * YOLO is fast and suitable for real-time object detection.
  * It performs detection in a single pass, making it efficient for complex scenes.

3. SSD (Single Shot MultiBox Detector) with TensorFlow
  * SSD is a real-time object detection method
  * SSD is efficient in terms of speed and accuracy.
  * Ideal for applications requiring both speed and moderate precision
    
4. SVM (Support Vector Machine)
  * A Support Vector Machine (SVM) is a powerful machine learning algorithm widely used for both linear and nonlinear classification.

### 0.1 Connect to Drive

```markdown
from google.colab import drive
drive.mount('/content/drive')
```

### 0.2 Installing Dependency

```markdown
!pip install ultralytics
```

### 0.3 Importing Libraries

```markdown
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import pandas as pd
import h5py
from os import listdir
from sklearn.metrics import accuracy_score
from ultralytics import YOLO
import tensorflow as tf
from sklearn.svm import SVC
import zipfile
import logging
import time
import os
from google.colab import files
```

## Exercise 1: HOG(Histogram of Oriented Gradients) Object Detection

```markdown
#Load an image
image = cv2.imread('/content/drive/MyDrive/photo.jpg')
gray_image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply HOG descriptor
features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

#Display the HOG image
plt.figuresize = (10, 10)
plt.axis('off')
plt.imshow(hog_image, cmap='gray')
plt.show()
```

![image](https://github.com/user-attachments/assets/1d2d34ea-81af-412e-9655-280faa72beb5)

## Exercise 2: YOLO (You Only Look Once) Object Detection

```markdown
# Load YOLO model configuration
net = cv2.dnn.readNet('/content/drive/MyDrive/yolov3.weights', '/content/drive/MyDrive/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
class_names = []
with open('/content/drive/MyDrive/coco.names', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# List of image paths to process
image_paths = [
    '/content/drive/MyDrive/image_detection.jpg',
    '/content/drive/MyDrive/folder_photos/image_detect2.jpeg',
    '/content/drive/MyDrive/folder_photos/image_detect3.jpeg'
]

# Output directory for processed images
output_dir = '/content/processed_images'
os.makedirs(output_dir, exist_ok=True)

# Process each image
for image_path in image_paths:
    # Load image
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detections in out:
            scores = detections[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Draw bounding box
                center_x = int(detections[0] * width)
                center_y = int(detections[1] * height)
                w = int(detections[2] * width)
                h = int(detections[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_names[class_id]} {confidence:.2f}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("YOLO Detection")
    plt.show()

      # Save the processed image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at: {output_path}")

    # Download the image (for Google Colab)
    files.download(output_path)
```

![image](https://github.com/user-attachments/assets/112318bf-9097-4b80-8853-95b33b1ffbd4)

## Exercise 3: SSD(Single Shot Multi Box Detection) with Tensorflow
### 3.1 Download SSD Model

```markdown
# Download the SSD MobileNet V2 COCO model
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
!tar -xzvf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
```

### 3.2 Load Images

```markdown
#Load pretrained SSD model
model = tf.saved_model.load("/content/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")

#Load image
image_path = "/content/drive/MyDrive/image_detection.jpg"
image_np = cv2.imread(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]
```

### 3.3 Run Detections

```markdown
#Run model detection
detections = model(input_tensor)

#Visualize the bounding boxes
for i in range(int(detections.pop("num_detections"))):
  if detections["detection_scores"][0][i] > 0.5:
    ymin, xmin, ymax, xmax = detections["detection_boxes"][0][i].numpy()
    (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                  ymin * image_np.shape[0], ymax * image_np.shape[0])

    #Draw bounding box
    cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

#Display the image
plt.figuresize = (10, 10)
plt.axis('off')
plt.title("SSD Detection")
plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.show()
```

## Exercise 4: Traditional vs. Deep Learning Object Detection Comparison
### 4.1 Extract Dataset

```markdown
import zipfile

#For Unzipping the file from google drive to colab cloud
zip_ref = zipfile.ZipFile("/content/drive/MyDrive/yolo_safety_dataset.zip", "r")
zip_ref.extractall("/content/dataset")
zip_ref.close()
```

### 4.2 Display Dataset Sample

```markdown
import os
import random
import matplotlib.pyplot as plt
import cv2

# Assuming train_image_path is defined in your code as "/content/dataset/css-data/train/images"
image_dir = "/content/dataset/css-data/train/images"

# Select 10 random image files
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
random_image_files = random.sample(image_files, min(10, len(image_files)))

# Display the images
plt.figure(figsize=(15, 10))
for i, image_file in enumerate(random_image_files):
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/2b02f929-9829-4695-bfd6-fdada2dfd58c)

### 4.3 Initializing Dataset path

```markdown
# Paths
train_image_path = "/content/dataset/css-data/train/images"
train_label_path = "/content/dataset/css-data/train/labels"
valid_image_path = "/content/dataset/css-data/valid/images"
valid_label_path = "/content/dataset/css-data/valid/labels"
test_image_path = "/content/dataset/css-data/test/images"
test_label_path = "/content/dataset/css-data/test/labels"
```

### 4.4 SVM Data Preprocessing

```markdown
#Get valid labels
def get_label(image_name, label_path):
    label_file = os.path.join(label_path, image_name.replace(".jpg", ".txt"))
    if not os.path.exists(label_file):
        print(f"Label file {label_file} does not exist. Skipping...")
        return None  

    with open(label_file, 'r') as file:
        label = file.read().strip()
    return label

#Preprocessing the image for training
def load_data(image_path, label_path):
    images, labels = [], []
    for img_name in os.listdir(image_path):
        img_path = os.path.join(image_path, img_name)

        #Applying grayslace
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Image {img_name} not loaded. Skipping...")
            continue

        #Image Resizing
        img = cv2.resize(img, (128, 128))
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        images.append(hog_features)

        label = get_label(img_name, label_path)
        if label is not None:  
            labels.append(label)
        else:
            print(f"No valid label found for {img_name}. Skipping...")

    return np.array(images), np.array(labels)
```

### 4.5 Data Splitting and Model Training

```markdown
# Load training, validation, and test data
train_X, train_y = load_data(train_image_path, train_label_path)
valid_X, valid_y = load_data(valid_image_path, valid_label_path)
test_X, test_y = load_data(test_image_path, test_label_path)

# Train the SVM classifier
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(train_X, train_y)
```

![image](https://github.com/user-attachments/assets/02c244dd-e3e4-4049-b53c-50ba2426489d)

### 4.6 SVM Validation

```markdown
# Validation performance
valid_predictions = svm_clf.predict(valid_X)
valid_accuracy = accuracy_score(valid_y, valid_predictions)
print("Validation Accuracy:", valid_accuracy)

# Test performance
test_predictions = svm_clf.predict(test_X)
test_accuracy = accuracy_score(test_y, test_predictions)
print("Test Accuracy:", test_accuracy)

# Optionally, timing the model for performance
import time
start_time = time.time()
test_predictions = svm_clf.predict(test_X)
end_time = time.time()
inference_time_per_image = (end_time - start_time) / len(test_X)
print("Inference Time per Image:", inference_time_per_image, "seconds")
```

![image](https://github.com/user-attachments/assets/e96c2d19-9311-4c02-bd77-15c3d32d3700)

### 4.7 Initializing YOLO model and path

```markdown
# Suppress Ultralytics logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load the YOLOv8 model using Ultralytics
yolo_model = YOLO('content/dataset/results_yolov8n_100e/kaggle/working/yolov8n.pt')

# Define paths
yolo_train_image_path = "/content/dataset/css-data/train/images"
yolo_train_label_path = "/content/dataset/css-data/train/labels"
yolo_valid_image_path = "/content/dataset/css-data/valid/images"
yolo_valid_label_path = "/content/dataset/css-data/valid/labels"
yolo_test_image_path = "/content/dataset/css-data/test/images"
yolo_test_label_path = "/content/dataset/css-data/test/labels"
```

### 4.8 YOLO data preprocessing

```markdown
#Get labels from text files
def yolo_get_label(image_name, label_path):
    label_file = os.path.join(label_path, image_name.replace(".jpg", ".txt"))
    if os.path.exists(label_file):
        with open(label_file, 'r') as file:
            lines = file.readlines()
            if lines:
                first_line = lines[0].strip().split()
                if first_line:
                    label = first_line[0]
                    return label
    return None

#Load images and labels 
def yolo_load_images_and_labels(image_path, label_path):
    yolo_images = []
    yolo_labels = []
    for img_name in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, img_name))
        if img is not None:
            label = yolo_get_label(img_name, label_path)
            if label is not None:
                yolo_images.append(img)
                yolo_labels.append(label)
    return yolo_images, yolo_labels

#Load and filter all data 
def yolo_load_all_data():
    yolo_train_images, yolo_train_labels = yolo_load_images_and_labels(yolo_train_image_path, yolo_train_label_path)
    yolo_valid_images, yolo_valid_labels = yolo_load_images_and_labels(yolo_valid_image_path, yolo_valid_label_path)
    yolo_test_images, yolo_test_labels = yolo_load_images_and_labels(yolo_test_image_path, yolo_test_label_path)
    return (yolo_train_images, yolo_train_labels), (yolo_valid_images, yolo_valid_labels), (yolo_test_images, yolo_test_labels)

# Load all data and filter out mismatches
(yolo_train_images, yolo_train_labels), (yolo_valid_images, yolo_valid_labels), (yolo_test_images, yolo_test_labels) = yolo_load_all_data()

# Function to evaluate the model
def yolo_evaluate_model(model, images, labels):
    yolo_all_predictions = []
    yolo_all_labels = []

    # Run inference
    for img, label in zip(images, labels):
        results = model(img, verbose=False)
        for result in results:
            boxes = result.boxes
            pred_classes = boxes.cls.cpu().numpy() if len(boxes) > 0 else []
            if len(pred_classes) > 0:
                yolo_all_predictions.append(int(pred_classes[0]))
                yolo_all_labels.append(int(float(label)))

    return yolo_all_labels, yolo_all_predictions
```

### 4.9 YOLO Validation

```markdown
# Validation performance
valid_labels, valid_predictions = yolo_evaluate_model(yolo_model, yolo_valid_images, yolo_valid_labels)
valid_accuracy = accuracy_score(valid_labels, valid_predictions)
print("Validation Accuracy:", valid_accuracy)

# Test performance
test_labels, test_predictions = yolo_evaluate_model(yolo_model, yolo_test_images, yolo_test_labels)
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", test_accuracy)

# Timing the model for performance on test set
start_time = time.time()
test_predictions = yolo_evaluate_model(yolo_model, yolo_test_images, yolo_test_labels)[1]  # Get predictions only
end_time = time.time()
inference_time_per_image = (end_time - start_time) / len(yolo_test_images)
print("Inference Time per Image:", inference_time_per_image, "seconds")
```

![image](https://github.com/user-attachments/assets/e1ca4f07-6c49-4053-9ca3-272026ecef76)

### 4.10 Performance Comparison

```markdown
# SVM with HOG performance
valid_predictions_svm = svm_clf.predict(valid_X)
valid_accuracy_svm = accuracy_score(valid_y, valid_predictions_svm)

test_predictions_svm = svm_clf.predict(test_X)
test_accuracy_svm = accuracy_score(test_y, test_predictions_svm)

# Timing the model for performance
import time
start_time = time.time()
_ = svm_clf.predict(test_X)
end_time = time.time()
inference_time_per_image_svm = (end_time - start_time) / len(test_X)

# YOLO performance
valid_labels_yolo, valid_predictions_yolo = yolo_evaluate_model(yolo_model, yolo_valid_images, yolo_valid_labels)
valid_accuracy_yolo = accuracy_score(valid_labels_yolo, valid_predictions_yolo)

test_labels_yolo, test_predictions_yolo = yolo_evaluate_model(yolo_model, yolo_test_images, yolo_test_labels)
test_accuracy_yolo = accuracy_score(test_labels_yolo, test_predictions_yolo)

# Timing the YOLO model for performance on test set
start_time = time.time()
_ = yolo_evaluate_model(yolo_model, yolo_test_images, yolo_test_labels)[1]  # Get predictions only
end_time = time.time()
inference_time_per_image_yolo = (end_time - start_time) / len(yolo_test_images)

# Print comparison side by side
print("\nPerformance Comparison:\n")
print(f"{'Model':<10} {'Validation Accuracy':<20} {'Test Accuracy':<20} {'Inference Time (s)':<20}")
print("-" * 70)
print(f"{'SVM with HOG':<10} {valid_accuracy_svm:<20.4f} {test_accuracy_svm:<20.4f} {inference_time_per_image_svm:<20.4f}")
print(f"{'YOLO':<10} {valid_accuracy_yolo:<20.4f} {test_accuracy_yolo:<20.4f} {inference_time_per_image_yolo:<20.4f}")
```

![image](https://github.com/user-attachments/assets/7d895737-87be-4f2a-8490-8fe9249d9a46)

