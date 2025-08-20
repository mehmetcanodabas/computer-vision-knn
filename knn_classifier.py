# Project: Implementation of K-Nearest Neighbors (KNN) for Image Classification
# Chair of Digital Signal Processing and Circuit Technology
# Computer Vision Laboratory
# Note: Source code is not shared due to institutional restrictions.
#       This file only contains architecture-level explanations.

import os
import argparse
from glob import glob
from collections import Counter
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# PART 1: DATASET LOADING
# Explanation:
# The dataset consists of images stored in class-based folders.
# Each folder corresponds to a category (e.g., Ball, Mug, Fork).
# All image paths are collected recursively to build the dataset.

# PART 2: IMAGE PREPROCESSING & FEATURE EXTRACTION
# Explanation:
# Two types of features are extracted:
#   (a) Color grid: image is downscaled to 4x4, normalized to [0,1],
#       flattened into 48 values (4*4*3).
#   (b) HOG (Histogram of Oriented Gradients): image is resized to 128x128,
#       gradients are computed with parameters:
#           - orientations = 12
#           - pixels_per_cell = (16,16)
#           - cells_per_block = (3,3)
# Final representation of each image is obtained by concatenating
# HOG features and color grid features.

# PART 3: KNN CLASSIFICATION
# Explanation:
# For each image, distances to all other samples are computed.
# The closest k neighbors are selected (Euclidean distance).
# Prediction = majority class of neighbors.
# Tie-breaking rule: choose the nearest neighbor’s label.

# PART 4: VISUALIZATION
# Explanation:
# A confusion matrix is generated and plotted as a heatmap.
# It shows how predictions are distributed across true classes,
# making it easier to spot which categories are misclassified.

# PART 5: EVALUATION
# Explanation:
# Classification performance is measured using:
#   - Accuracy
#   - Precision (macro average)
#   - Recall (macro average)
# A confusion matrix is also computed for detailed error analysis.

 