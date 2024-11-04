# 4B-SANCON-MP
## Sancon, Simon B - BSCS4B

## Important Links
1. [Actual Notebook](https://colab.research.google.com/drive/1rrdgXE52Q9eUhjitGpq8t4xQCgBHm0Uv?usp=sharing)
2. [Repository Notebook](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP/code/4B_SANCON_MP.ipynb)
3. [Image Folder](https://github.com/sancon-simon/CSST106-4B/tree/main/4B-SANCON-MP/images)
4. [Video Presentation](https://github.com/sancon-simon/CSST106-4B/tree/main/4B-SANCON-MP/video)
5. [Pascal Dataset](https://public.roboflow.com/object-detection/pascal-voc-2012)

## Hands On Exploration

### 1. Selection of Dataset and Algorithm:
   
This Mid-Term Project includes implementing object detection on dataset
The Algorithm that was used is YOLOv5 and the dataset utilized was Pascal VOC 2012 Dataset
The dataset is composed of 17112 labeled images in roboflow the splitting was done after labeling
The algorithm was selected for computing constraints reasons and compatibility for better evaluation.

![image](https://github.com/user-attachments/assets/0f0210aa-5e56-4eb4-8e94-8d5d38f89cb9)

### 2. Implementation:

2.1 Data Preparation:

The data was preprocessed using rescaling and grayscaling methods utilizing cv2

```markdown
#Load Image
train_path = '/content/dataset2/train/images'
valid_path = '/content/dataset2/valid/images'

train_label_path = '/content/dataset2/train/labels'
valid_label_path = '/content/dataset2/valid/labels'

#Preprocess Image
def preprocess_images(image_dir, output_dir, size=(640, 640)):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
      img_path = os.path.join(image_dir, filename)
      img = cv2.imread(img_path)
      if img is not None:
        img = cv2.resize(img, size) # Resize
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Grayscale
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img_gray)


# Preprocess training images
preprocess_images(train_path, '/content/preprocessed_train')

# Preprocess validation images
preprocess_images(valid_path, '/content/preprocessed_valid')
```

![image](https://github.com/user-attachments/assets/43722f3f-8820-4def-8ea5-9461fce04f96)

2.2 Model Building:

Pytorch was utilized to load the pretrained yolov5 model.
The neural network of yolov5 is better explained in the vidio above.

```markdown
#Load YOLO Model
import torch

# Load a pretrained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Ultralytics YOLOv5 🚀, AGPL-3.0 license
'''
# Parameters
nc: 80 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]], # 9
  ]

# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
'''
```

2.3 Training the Model:

The was trained using a split of training and validation of pascal dataset
The split was 80-20 percent ratio with numbers 13690 to 3422

2.4 Testing:

In this section the performance of the trained is tested in detecting objects.

![image](https://github.com/user-attachments/assets/9933b6f1-1aa2-4b6c-a8c2-f3ec3e821aeb)

### 3. Evaluation

In the evaluation section ssd model was the model selected for comparison.
The YOLO and SSD model was evaluated with the same dataset further preprocessing was done for evaluation accuracy.

SSD evaluation metrics:

![image](https://github.com/user-attachments/assets/40832bba-8827-46e8-8d2e-fc1885462068)

YOLO evaluation metrics:

![image](https://github.com/user-attachments/assets/b4259cb3-a18c-48a3-bd23-6a0993d1ce52)







