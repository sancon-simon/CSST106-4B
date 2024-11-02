#Importing Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
