# Automated Quiz Checking with Computer Vision

A web-based application designed to automate the evaluation of multiple-choice quiz sheets using computer vision.

## Project Overview
The system leverages the OpenCV library to process images of answer sheets. The workflow includes:
1. Perspective transformation to align and flatten the document.
2. Image binarization (thresholding) to isolate marked answers.
3. Grid analysis to detect selected bubbles and map them to correct answer keys.
4. Calculation of the final score and recognition accuracy.

## Demo Video
Project demonstration: [https://www.youtube.com/embed/6z8OhLEcITE](https://www.youtube.com/embed/6z8OhLEcITE)

## Installation and Setup

### Prerequisites
* Python 3.8+
* Dependencies listed in `requirements.txt`
  
### You can run by this command
* python app.py

### Link for website
* https://automated-blank-checking.onrender.com
