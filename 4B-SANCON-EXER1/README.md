# 4B-SANCON-EXER1
## Sancon, Simon B - BSCS4B

## Colab Notebook
https://colab.research.google.com/drive/1ra8S-dnOLtrFGJsRjb7BpxBtTyYQhk7F?usp=sharing

## Hands-On Exploration:
### 1. Install OpenCV

```markdown
!pip install opencv-python-headless
```

### 2. Import Libraries and Function Definition

```markdown
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(img, title="Image"):
  plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.axis("off")
  plt.show()

def display_images(img1, img2, title1="Image 1", title2="Image 2"):
  plt.subplot(1,2,1)
  plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
  plt.title(title1)
  plt.axis("off")

  plt.subplot(1,2,2)
  plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
  plt.title(title2)
  plt.axis("off")
  plt.show()
```

### 3. Load Image

```markdown
from google.colab import drive
drive.mount('/content/drive')

image_path = '/content/drive/MyDrive/photo.jpg'  # Replace with your image path
image = cv2.imread(image_path)
display_image(image, "Original Image")
```
![image](https://github.com/user-attachments/assets/55731e34-de9b-4009-a9ce-208c1e6f879e)

```markdown
'''
#for uploading from local
from google.colab import files
from io import BytesIO
from PIL import Image

uploaded = files.upload()
image_path = next(iter(uploaded))
image = Image.open(BytesIO(uploaded[image_path]))
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

display_image(image,"Original Image")
'''
```

### 1. Exercise 1: Scaling and Orientation

```markdown
def scale_image(image, scale_factor):
  height, width = image.shape[:2]
  scale_img = cv2.resize(image,(int(width * scale_factor), int(height * scale_factor)), interpolation = cv2.INTER_LINEAR)
  return scale_img

def rotate_image(image, angle):
  height, width = image.shape[:2]
  center = (width//2,height//2)
  matrix = cv2.getRotationMatrix2D(center,angle,1)
  rotated_image = cv2.warpAffine(image,matrix,(width,height))
  return rotated_image

scaled_image = scale_image(image, 0.5)
display_image(scaled_image,"Scaled Image")

rotated_image = rotate_image(image, 45)
display_image(rotated_image,"Rotated Image")
```
![image](https://github.com/user-attachments/assets/aba979a5-9e0e-44d2-9387-7fc923c48ffb)  ![image](https://github.com/user-attachments/assets/9fa97849-c50b-45ab-ac3b-cef96f0523b3)

