# 4B-SANCON-EXER2
## Sancon, Simon B - BSCS4B

## Colab Notebook
[4B-SANCON-ECER2.ipynb](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-EXER2/code/4B_SANCON_EXER2.ipynb)

## Task 1: SIFT Feature Extraction
### 1.1 Connecting to Google Drive

```markdown
from google.colab import drive
drive.mount('/content/drive')
```

### 1.2 Importing Libraries

```markdown
import cv2
import matplotlib.pyplot as plt
```

### 1.3 Apply SIFT with the selected image

```markdown
#Load Image
image = cv2.imread("/content/drive/MyDrive/photo.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Initialize SIFT Detector
sift = cv2.SIFT_create()

#Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

#Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

#Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("SIFT Keypoints")
plt.show()
```
![image](https://github.com/user-attachments/assets/d512ec06-3f02-488b-a94f-4a4baaa1445f)

## Task 2: SURF Feature Extraction
### 2.1 Updating OpenCV to make SURF work

```markdown
!pip uninstall opencv-python opencv-contrib-python
!apt-get update
!apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

!git clone https://github.com/opencv/opencv.git
!git clone https://github.com/opencv/opencv_contrib.git
import os
os.makedirs('opencv/build', exist_ok=True)
os.chdir('opencv/build')

!cmake -D CMAKE_BUILD_TYPE=Release \
       -D CMAKE_INSTALL_PREFIX=/usr/local \
       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
       -D OPENCV_ENABLE_NONFREE=ON \
       ..

!make -j$(nproc)
!make install

!pip install opencv-python opencv-python-headless opencv-contrib-python-headless

```

### 2.2 Loading Image

```markdown
#Load Images
image = cv2.imread("/content/drive/MyDrive/photo.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 2.3 Initialize SURF and Visualize Output

```markdown
#Initialize SURF
surf = cv2.xfeatures2d.SURF_create()

#Detect keypoints and descriptors
keypoints, descriptors = surf.detectAndCompute(gray_image, None)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

#Draw keypoints
image_with_keypoints = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

#Plot the image
plt.imshow(image_with_keypoints)
plt.axis("off")
plt.title("SURF Keypoints")
plt.show()
```

![image](https://github.com/user-attachments/assets/167848e5-bbdd-4a08-b5e8-7ab469e522a4)


## Task 3: Applying ORB and Visualize Image
### 3.1 Loading Image

```markdown
#same image ---> image = cv2.imread("/content/drive/MyDrive/photo.jpg")
#samge gray_image ---> gray_image = cv2.Color(image, cv2.COLOR_BGR2GRAY)
```

### 3.2 Applying ORB and Visualizing Image

```markdown
#Initialize ORB
orb = cv2.ORB_create()

#Detect keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

#Draw keypoints
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

#Plot the image
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("ORB Keypoints")
plt.show()
```

![image](https://github.com/user-attachments/assets/243ed8cd-14c4-4299-87fc-32027969894e)

## Task 4: Feature Matching 
### 4.1 Feature Matching with SIFT

```markdown
# Load two images
image1 = cv2.imread("/content/drive/MyDrive/photo.jpg")
image2 = cv2.imread("/content/drive/MyDrive/rotated_image.jpg")

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Initialize the matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(image_matches)
plt.axis("off")
plt.title("Feature Matching with SIFT")
plt.show()
```

![image](https://github.com/user-attachments/assets/0cd34a42-f879-4c29-b29e-2bf442f1922d)

## Task 5: Application of Feature Matching
### 5.1 Image Alignment Using Homography

```markdown
import numpy as np

image1 = cv2.imread("/content/drive/MyDrive/rotated_image.jpg")
image2 = cv2.imread("/content/drive/MyDrive/photo.jpg")

# Convert to Grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Image features using BFmatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract location of good matches
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# Find homograpy matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp one image to align with the other
h, w, _= image1.shape
result = cv2.warpPerspective(image1, M, (w, h))

# Display both the original and warped images side by side
plt.figure(figsize=(20, 10))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Original Image")

# Warped image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Image Alignment using Homography")

plt.show()
```

![image](https://github.com/user-attachments/assets/ce1e8ba0-a352-4341-97a9-f43fbc5d7fd3)


## Task 6: Combining Feature Extraction Methods
### 6.1 SIFT and ORB Feature Matching

```markdown
# Load images
image1 = cv2.imread("/content/drive/MyDrive/photo.jpg")
image2 = cv2.imread("/content/drive/MyDrive/rotated_image.jpg")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# SIFT detector
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(gray1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(gray2, None)

# ORB detector
orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(image1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(image2, None)

# BFMatcher for SIFT (using cv2.NORM_L2)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_sift.match(descriptors1_sift, descriptors2_sift)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)  # Sort matches by distance

# BFMatcher for ORB (using cv2.NORM_HAMMING)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(descriptors1_orb, descriptors2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# Draw the top matches for SIFT
image_sift_matches = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, matches_sift[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw the top matches for ORB
image_orb_matches = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, matches_orb[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(image_sift_matches, cv2.COLOR_BGR2RGB))
plt.title('SIFT Matches')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(image_orb_matches, cv2.COLOR_BGR2RGB))
plt.title('ORB Matches')
plt.axis('off')

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/75e1ab50-9caf-452a-bda9-84663686ddc2)

![image](https://github.com/user-attachments/assets/aeb5ada8-af5c-42f4-ade3-0b2fea49d101)




