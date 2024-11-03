#Importing Libraries
import cv2
import os
import matplotlib.pyplot as plt

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
