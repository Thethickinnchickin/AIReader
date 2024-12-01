import os
import pandas as pd

# Path to the dataset
data_dir = "CUB_200_2011/"

# Load metadata
image_paths = pd.read_csv(os.path.join(data_dir, "images.txt"), sep=" ", header=None, names=["image_id", "file_path"])
class_labels = pd.read_csv(os.path.join(data_dir, "image_class_labels.txt"), sep=" ", header=None, names=["image_id", "class_id"])
bounding_boxes = pd.read_csv(os.path.join(data_dir, "bounding_boxes.txt"), sep=" ", header=None, names=["image_id", "x", "y", "width", "height"])
train_test_split = pd.read_csv(os.path.join(data_dir, "train_test_split.txt"), sep=" ", header=None, names=["image_id", "is_training"])

# Merge metadata
metadata = image_paths.merge(class_labels, on="image_id").merge(bounding_boxes, on="image_id").merge(train_test_split, on="image_id")
print(metadata.head())
