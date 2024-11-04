# 4B-SANCON-MP5
## Sancon, Simon B.

## Important Links
1. [Actual Notebook](https://colab.research.google.com/drive/1lCwUBScOdIkUU6DMHGuWE6ZLwNMnXMpX?usp=sharing)
2. [Repository Notebook](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP5/code/4B_SANCON_MP5.ipynb)
3. [Image Folder](https://github.com/sancon-simon/CSST106-4B/tree/main/4B-SANCON-MP5/images)

## Hands-On Exploration

This machine problem involves utilizing yolo model and ssd model to conduct object detection and recognition methods
The model was tested on at least three different images to compare its performance and observe its accuracy
The model weights was downloaded and was convert to accomodate tensorflow format

Step zeros 
  * Connect to Drive
  * Installing Dependencies
  * Downloading YOLO Model Weights and Converting to TensorFlow Format
  * Importing Libraries

1. Load YOLO Model Using TensorFlow
2. Image Input
3. Object Detection
4. Visualization
5. Testing
6. Performance Analysis
   
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

[Performance Analysis](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP5/performance-analysis/4B-SANCON-MP5-performance-analysis.md)
