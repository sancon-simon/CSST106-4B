#Importing Libraries
import cv2
import os
import time
import matplotlib.pyplot as plt

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

# Save ipath
save_path = '/content/drive/MyDrive/FeatureMatchingResults'  # Change to your desired path in Google Drive
os.makedirs(save_path, exist_ok=True)

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
