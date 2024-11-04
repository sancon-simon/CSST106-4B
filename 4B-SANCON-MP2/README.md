# Machine Problem 2: Image Processing Techniques 
## Sancon, Simon B - BSCS4B

[Colab Notebook MP2](https://colab.research.google.com/drive/1tghjIupyJzLcixliZBXQGbywJ5YWPPdx?usp=sharing)
[Colab Notevook Exer1](https://colab.research.google.com/drive/1ra8S-dnOLtrFGJsRjb7BpxBtTyYQhk7F?usp=sharing)

The notebook here was originally EXER1,
In here after we did EXER 1, next was the MP2 but,
The EXER 1 has an additional activity later on thats why MP2 was kinda overwritten

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

### 4. Exercise 1: Scaling and Orientation

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

### 5. Exercise 2: Blurring Techniques

```markdown
guassian_blur = cv2.GaussianBlur(image,(61,61),0)
display_image(guassian_blur,"Guassian Blur")

median_blur = cv2.medianBlur(image,31)
display_image(median_blur,"Median Blur")

bilateral_blur = cv2.bilateralFilter(image,99,75,75)
display_image(bilateral_blur,"Bilateral Blur")
```

### 6.  Edge Detection using Canny

```markdown
edge = cv2.Canny(image,75 ,150)
display_image(edge, "Canny Edge Detection")
```

## Problem-Solving Session
### Common challenges
There are 2 common challenges that was experienced during the Problem-Solving Session

1. Syntax error
   
![image](https://github.com/user-attachments/assets/a0d076c8-9f0c-4673-a564-b4ba3dedf85b)

The figure above is one common issue that will be tackled whenever an implementation was conducted, this includes spelling, misused variables, and unfamiliarity of syntax or library.


2. Finding the right value
   
![image](https://github.com/user-attachments/assets/4186a81f-3177-4981-b91c-de8c40505434)

The figure above shows that finding the right value of the applied image processing techniques requires some trial and error to produce the desired result.

### Scenario-Based Problems: 

![image](https://github.com/user-attachments/assets/9683cbee-0aa3-4e46-afcb-ddd08b3fc1c3)

The figure above shows that we can apply Blurring to enhance a photo to reduce noise from the photo



