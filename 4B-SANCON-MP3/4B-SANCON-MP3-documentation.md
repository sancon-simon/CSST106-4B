# 4B-SANCON-MP3
## Sancon, Simon B - BSCS4B

## Important Links
1. [Actual Notebook](https://colab.research.google.com/drive/1o-BoWWtT2daByHihTrGOgNoy4QFE5Zv2?usp=sharing)
2. [Repository Notebook](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP3/code/4B_SANCON_MP3.ipynb)
3. [Feature_extraction.py](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP3/code/feature_extraction.py)
4. [Feature_matching.py](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP3/code/feature_matching.py)
5. [Image_alignment.py](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP3/code/image_alignment.py)
6. [Image_folder](https://github.com/sancon-simon/CSST106-4B/tree/main/4B-SANCON-MP3/images)
7. [Performance Analysis](https://github.com/sancon-simon/CSST106-4B/blob/main/4B-SANCON-MP3/performance_analysis/performance_analysis.md)

## Hands-On Exploration:

The First following steps zeros includes environment preparation
encompassing updating opencv environment to make surf work,
importing libraries, initializing calculation functions and save paths

### Step 0.1 Updating Open-CV

```markdown
from google.colab import drive
drive.mount('/content/drive')

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

```

### Step 0.2 Importing Libraries

```markdown
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
```

### Step 0.3 Initializing accuracy calculation

```markdown
# Lowe's ratio test function for feature matching accuracy
def calculate_accuracy(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    accuracy = len(good_matches) / len(matches) if len(matches) > 0 else 0
    return accuracy, len(good_matches)
```

### Step 0.4 Initializing save paths

```markdown
# Save ipath
save_path = '/content/drive/MyDrive/FeatureMatchingResults'  # Change to your desired path in Google Drive
os.makedirs(save_path, exist_ok=True)
```

## Step 1. Loading Images
This step is conducted at the start of each cells/python code

## Step 2. Extract Keypoints and Descriptors Using SIFT, SURF and ORB

This Section includes conduction feature extraction and matching 
The feature extraction utilizes SIFT, SURF, and ORB
Two images was used with different orientation of book perspective

### Step 2.1 SIFT Feature Matching

```markdown
#Load Image
image1 = cv2.imread("/content/drive/MyDrive/image11_2.png")
image2 = cv2.imread("/content/drive/MyDrive/image11_3.png")

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#Initialize SIFT Detector
sift = cv2.SIFT_create()

#Detect keypoints and descriptors
start_time = time.time()
keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(gray_image1, None)
keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(gray_image2, None)
sift_time = time.time() - start_time
accuracy_sift, good_matches_sift = calculate_accuracy(descriptors_sift1, descriptors_sift2)
print(f"SIFT detection time: {sift_time:.4f} seconds")
print(f"SIFT number of keypoints: {len(keypoints_sift1)} and {len(keypoints_sift2)}")
print(f"SIFT accuracy (good matches ratio): {accuracy_sift:.4f}, good matches: {good_matches_sift}")

#Draw keypoints on the image
image_sift1_with_keypoints = cv2.drawKeypoints(image1, keypoints_sift1, None)
image_sift2_with_keypoints = cv2.drawKeypoints(image2, keypoints_sift2, None)

#Display the image with keypoints
plt.figuresize = (10, 10)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_sift1_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("SIFT Keypoints")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_sift2_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("SIFT Keypoints")
plt.show()

#Save SIFT images
cv2.imwrite(os.path.join(save_path, 'sift_keypoints_image1.jpg'), image_sift1_with_keypoints)
cv2.imwrite(os.path.join(save_path, 'sift_keypoints_image2.jpg'), image_sift2_with_keypoints)
```

![image](https://github.com/user-attachments/assets/6454085d-a3fc-4b8d-b9e6-f6f4bd528032)

### Step 2.2 SURF Feature Matching

```markdown
#Load Image
image1 = cv2.imread("/content/drive/MyDrive/image11_2.png")
image2 = cv2.imread("/content/drive/MyDrive/image11_3.png")

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()

#Detect keypoints and descriptors
start_time = time.time()
keypoints_surf1, descriptors_surf1 = surf.detectAndCompute(gray_image1, None)
keypoints_surf2, descriptors_surf2 = surf.detectAndCompute(gray_image2, None)
surf_time = time.time() - start_time
accuracy_surf, good_matches_surf = calculate_accuracy(descriptors_surf1, descriptors_surf2)
print(f"SURF detection time: {surf_time:.4f} seconds")
print(f"SURF number of keypoints: {len(keypoints_surf1)} and {len(keypoints_surf2)}")
print(f"SURF accuracy (good matches ratio): {accuracy_surf:.4f}, good matches: {good_matches_surf}")

#Draw keypoints on the image
image_surf1_with_keypoints = cv2.drawKeypoints(image1, keypoints_surf1, None)
image_surf2_with_keypoints = cv2.drawKeypoints(image2, keypoints_surf2, None)

#Display the image with keypoints
plt.figuresize = (10, 10)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_surf1_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("SURF Keypoints")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_surf2_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("SURF Keypoints")
plt.show()

#Save SURF images
cv2.imwrite(os.path.join(save_path, 'surf_keypoints_image1.jpg'), image_surf1_with_keypoints)
cv2.imwrite(os.path.join(save_path, 'surf_keypoints_image2.jpg'), image_surf2_with_keypoints)
```

![image](https://github.com/user-attachments/assets/d152fb9f-3000-4941-a06a-be8734dc0510)

### Step 2.3 ORB Feature Matching

```markdown
#Load Image
image1 = cv2.imread("/content/drive/MyDrive/image11_2.png")
image2 = cv2.imread("/content/drive/MyDrive/image11_3.png")

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

#Detect keypoints and descriptors
start_time = time.time()
keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(gray_image1, None)
keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(gray_image2, None)
orb_time = time.time() - start_time
accuracy_orb, good_matches_orb = calculate_accuracy(descriptors_orb1, descriptors_orb2)
print(f"ORB detection time: {orb_time:.4f} seconds")
print(f"ORB number of keypoints: {len(keypoints_orb1)} and {len(keypoints_orb2)}")
print(f"ORB accuracy (good matches ratio): {accuracy_orb:.4f}, good matches: {good_matches_orb}")

#Draw keypoints on the image
image_orb1_with_keypoints = cv2.drawKeypoints(image1, keypoints_orb1, None)
image_orb2_with_keypoints = cv2.drawKeypoints(image2, keypoints_orb2, None)

#Display the image with keypoints
plt.figuresize = (10, 10)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_orb1_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("ORB Keypoints")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_orb2_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("ORB Keypoints")
plt.show()

#Save ORB images
cv2.imwrite(os.path.join(save_path, 'orb_keypoints_image1.jpg'), image_orb1_with_keypoints)
cv2.imwrite(os.path.join(save_path, 'orb_keypoints_image2.jpg'), image_orb2_with_keypoints)
```

![image](https://github.com/user-attachments/assets/112c4d34-0757-4234-8439-c86caa73c01e)

## Step 3. Feature Matching with Brute-Force and FLANN

This section include conducting Feature Matching
The Matching Utilizes Brute-Force and FLANN

### Step 3.1 Feature Matching Using Brute-Force

```markdown
# Load Images
image1 = cv2.imread("/content/drive/MyDrive/image11_2.png")
image2 = cv2.imread("/content/drive/MyDrive/image11_3.png")

# Resize the second image to match the size of the first image
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Convert to grayscale for feature detection
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2_resized = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints_bf1, descriptors_bf1 = sift.detectAndCompute(gray_image1, None)
keypoints_bf2, descriptors_bf2 = sift.detectAndCompute(gray_image2_resized, None)

# Initialize the Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors_bf1, descriptors_bf2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches using the original color images, not grayscale
image_matches = cv2.drawMatches(image1, keypoints_bf1, image2_resized, keypoints_bf2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.imshow(cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Feature Matching with SIFT (Brute-Force Matcher)")
plt.show()

#Save image
cv2.imwrite(os.path.join(save_path, 'sift_bf_match.jpg'), image_matches)
```

![image](https://github.com/user-attachments/assets/517a3185-1d6a-417e-9ca0-2f166d6a5311)

### Step 3.2 Feature Matching Using FLANN

```markdown
# Load Images
image1 = cv2.imread("/content/drive/MyDrive/image11_2.png")
image2 = cv2.imread("/content/drive/MyDrive/image11_3.png")

# Resize the second image to match the size of the first image
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Convert to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2_resized = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints_flann1, descriptors_flann1 = sift.detectAndCompute(gray_image1, None)
keypoints_flann2, descriptors_flann2 = sift.detectAndCompute(gray_image2_resized, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

f_matches = flann.knnMatch(descriptors_flann1, descriptors_flann2, k=2)

# Apply ratio test
good_matches = []
for m, n in f_matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw matches
image_matches = cv2.drawMatches(image1, keypoints_flann1, image2_resized, keypoints_flann2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.imshow(cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Feature Matching with SIFT and FLANN (Resized Images)")
plt.show()

#Save image
cv2.imwrite(os.path.join(save_path, 'sift_flann_match.jpg'), image_matches)
```

![image](https://github.com/user-attachments/assets/dde7456a-d8a4-4bae-ab85-d6cf99adcdff)

## Step 4. Image Alignment using Homography

This Section includes conduction Image alignment using Homography
In here the image was warped to match the image 1 orientation or perspective

```markdown
import numpy as np

# Load Images
image1 = cv2.imread("/content/drive/MyDrive/image11_2.png")
image2 = cv2.imread("/content/drive/MyDrive/image11_3.png")

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
src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)

# Find homograpy matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp one image to align with the other
h, w, _= image1.shape
result = cv2.warpPerspective(image2, M, (w, h))

# Display both the original and warped images side by side
plt.figure(figsize=(20, 10))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Image 1")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Image 2(Warped)")

# Warped image
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Aligning Image2 using Homography")

plt.show()

#Save image
cv2.imwrite(os.path.join(save_path, 'aligned_image.jpg'), image2)
cv2.imwrite(os.path.join(save_path, 'warped_image.jpg'), result)
```

![image](https://github.com/user-attachments/assets/54d6875d-130a-4698-b631-1201cab0fa1f)

## Step 5. Performance Analysis

This Section includes the Results and Performance of SIFT, SURF and ORB
This includes metrics such as Accuracy, Keypoints Detected, and Time of Feature Extraction

### Step 5.1 Comapre the Results

```markdown
data = {
    'Detector': ['SIFT', 'SURF', 'ORB'],
    'Accuracy': [accuracy_sift, accuracy_surf, accuracy_orb],
    'Keypoints': [len(keypoints_sift1), len(keypoints_surf1), len(keypoints_orb1)],
    'Time of feature extraction (s)': [sift_time, surf_time, orb_time]
}
df = pd.DataFrame(data)
print(df)
```

![image](https://github.com/user-attachments/assets/57748efa-d0b3-4e51-a8cb-36e95d8813d7)

### Step 5.2 Write a Short Report

```markdown
print('''
The Brute-Force Matcher can be compared with the FLANN Matcher in terms of accuracy
Based on the visualization both produce minimal mistake but BF matcher
made a mistake of features between the book and the table. In comparison to that
FLANN Matcher only made a mistake of matching the corner of the book.
''')
```

![image](https://github.com/user-attachments/assets/b3f1a58f-2d4a-4d75-94df-ad495494f953)


