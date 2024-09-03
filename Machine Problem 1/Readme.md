# Exploring the Role of Computer Vision and Image Processing in AI
## Sancon, Simon B - BSCS4B

## Introduction to Computer Vision and Image Processing
Computer vision is a field of AI that uses machine learning and neural networks to help computers interpret and derive meaningful information from digital images, videos, and other visual inputs. It builds on image processing techniques to analyze visual data, make recommendations, and take actions based on detected patterns or defects.

Image processing serves as a means of translation between the human visual system and digital imaging devices. The human visual system does not perceive the world in the same manner as digital detectors, with display devices imposing additional noise and bandwidth restrictions.

## Types of Image Processing Techniques

### Filtering
![Screenshot 2024-09-03 140642](https://github.com/user-attachments/assets/858251c3-9898-42a7-ad6c-b2cf90bf3b1e)

The goal of using filters is to modify or enhance image properties and/or to extract valuable information from the pictures such as edges, corners, and blobs. 

### Edge Detection

![Screenshot 2024-09-03 140722](https://github.com/user-attachments/assets/d3a356da-688a-4503-b2ba-bda1584be759)

The goal of edge detection is to identify the most significant edges within an image or scene. These detected edges should be connected to form meaningful lines and boundaries.

### Segmentation

![Screenshot 2024-09-03 140852](https://github.com/user-attachments/assets/c084a0eb-9c91-43c1-9c75-a15d27a1c5c1)

The goal of  segmentation  is to simplify and analyze images by separating them into different segments. This makes it easier for computers to understand the content of the image.

## Case Study Overview

### Application

An OCR program extracts and repurposes data from scanned documents, camera images and image-only PDFs. It singles out letters on the image, puts them into words, and then puts the words into sentences, thus enabling access to and editing of the original content. It also eliminates the wasted effort of redundant manual data entry.

### Challenges

OCR software’s performance hinges on the quality of the source images or documents. Low-resolution images, faded text, or poor lighting conditions .this can be solved using Median filter by applying blur to noisy images while preserving the edges of high-contrast objects like letters with the help of binarization.

## Image Processing Implementation

| Image 1                  | Image 2                  |
|--------------------------|--------------------------|
| ![Image 1](![Screenshot 2024-09-03 142557](https://github.com/user-attachments/assets/cd996441-ffb3-479a-87bb-ff511fb52b5d))   | ![Image 2](![Screenshot 2024-09-03 142622](https://github.com/user-attachments/assets/0667812f-de82-4a93-8446-1d9b552ee787))   |


