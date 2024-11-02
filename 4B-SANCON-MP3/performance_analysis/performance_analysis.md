# Step 5: Performance Analysis

This Section includes the analysis of the performance of SIFT, SURF, and ORB in terms of keypoint detection accuracy, number of keypoints detected, and speed. Additionaly this contains a report on the effectiveness of Brute-Force Matcher versus FLANN Matcher for feature matching.

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

![image](https://github.com/user-attachments/assets/0b848f07-670e-4176-8af1-d1540eeb999d)

### Step 5.2 Write a Short Report

```markdown
print('''
The Brute-Force Matcher can be compared with the FLANN Matcher in terms of accuracy
Based on the visualization both produce minimal mistake but BF matcher
made a mistake of features between the book and the table. In comparison to that
FLANN Matcher only made a mistake of matching the corner of the book.
''')
```

![image](https://github.com/user-attachments/assets/cdd1bdc3-fa60-423d-9b03-06507a9ae33b)

