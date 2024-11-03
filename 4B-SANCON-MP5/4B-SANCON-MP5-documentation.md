# 4B-SANCON-MP5
## Sancon, Simon B.

## Important Links
1. [Actual Notebook]()
2. [Repository Notebook]()
3. [Image Folder]()

## Hands-On Exploration

0.1 Connect to Drive
  * This step involves mounting Google Drive to access files stored there.
0.2 Installing Dependencies
  * In this step, the YOLOv3 TensorFlow implementation is cloned from GitHub, and the required Python packages are installed using a requirements file.
0.3 Downloading YOLO Model Weights and Converting to TensorFlow Format
  * This section downloads the pre-trained YOLOv3 weights from a specified URL and converts them into a TensorFlow-compatible format using a conversion script.
0.4 Importing Libraries
  * Essential libraries for image processing, numerical operations, and TensorFlow functionalities are imported. This includes libraries like OpenCV, NumPy, TensorFlow, and others necessary for running the YOLO model.
1. Load YOLO Model Using TensorFlow
  * In this part, command-line flags are parsed, parameters such as class names and weights paths are defined, the YOLO model is instantiated with the number of classes, and the model weights are loaded. The class names are also read from a file.
2. Image Input
  * Here, paths to images that will be processed for object detection are initialized.
3. Object Detection
  * This section describes how to read an image, preprocess it for the model, and run inference to detect objects within the image.
4. Visualization
  * The inference time is printed, and detected objects along with their confidence scores and bounding box coordinates are displayed. The results are visualized on the original image using Matplotlib.
5. Testing
  * In this step, each image in the previously defined list is processed for object detection in a loop, where inference times and detection results are printed and visualized for each image.
6. Performance Analysis
  * This section would typically involve analyzing the performance of the object detection process in terms of speed (inference time) and accuracy (detection results). This summary captures the key components of your hands-on exploration without including any code snippets.
    
### 0.1 Connect to Drive

```markdown
from google.colab import drive
drive.mount('/content/drive'
```
### 0.2 Installing Dependencies

```markdown
!git clone https://github.com/zzh8829/yolov3-tf2.git
cd yolov3-tf2
!pip install -r requirements.txt
```

### 0.3 Downloading YOLO model weigths and converting to tensorflow formal

```markdown
!wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
!python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

### 0.4 Importing Libraries

```markdown
import time
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from absl import app, flags
import matplotlib.pyplot as plt
```

## 1. Load YOLO model using Tensorflow

```markdown
#Parse flags
FLAGS = flags.FLAGS
FLAGS(argv=['']) 

#Define parameters
classes_path = './data/coco.names'
weights_path = './checkpoints/yolov3.tf'

#Resize images 
size = 416  

#Load model and weights
yolo = YoloV3(classes=80)
yolo.load_weights(weights_path).expect_partial()

#Load class names
class_names = [c.strip() for c in open(classes_path).readlines()]
```

## 2. Image Input

```markdown
#Initialize image paths
image_paths = [
    '/content/drive/MyDrive/folder_photos/image_detect1.jpeg',
    '/content/drive/MyDrive/folder_photos/image_detect2.jpeg',
    '/content/drive/MyDrive/folder_photos/image_detect3.jpeg'
]
```

## 3. Object Detection

```markdown
# Specify the single image path
image_path = '/content/drive/MyDrive/folder_photos/image_detect1.jpeg'

# Read and process the image
img_raw = cv2.imread(image_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

img = tf.expand_dims(img_raw, 0)
img = transform_images(img, size)

# Run inference
t1 = time.time()
boxes, scores, classes, nums = yolo(img)
t2 = time.time()
```

## 4. Visualization

```markdown
#Print inference time
print(f'Time taken for {image_path}: {t2 - t1:.4f} seconds')

# Display detections
print('Detections for {}:'.format(image_path))
for i in range(nums[0]):
    print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                 np.array(scores[0][i]),
                                 np.array(boxes[0][i])))

# Draw outputs on the image
img_output = draw_outputs(img_raw, (boxes, scores, classes, nums), class_names)

# Display the output using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(img_output)
plt.axis('off')  # Hide axes for a cleaner look
plt.title('Detections for {}'.format(image_path.split("/")[-1]))
plt.show()  # Show the image without saving it
```

## 5. Testing

```markdown
# Process each image for inference
for image_path in image_paths:
    img_raw = cv2.imread(image_path)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    # Run inference
    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print(f'Time taken for {image_path}: {t2 - t1:.4f} seconds')

    # Display detections
    print('Detections for {}:'.format(image_path))
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                     np.array(scores[0][i]),
                                     np.array(boxes[0][i])))

    # Draw outputs on the image
    img_output = draw_outputs(img_raw, (boxes, scores, classes, nums), class_names)

    # Display the output using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_output)
    plt.axis('off')  # Hide axes
    plt.title(f'Detections for {image_path.split("/")[-1]}')
    plt.show()  # Show the image without saving it
```

## 6. Performance Analysis
