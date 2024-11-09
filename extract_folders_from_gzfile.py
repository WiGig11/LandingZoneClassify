from PIL import Image
import struct
import os
import cv2
from tqdm import tqdm
import string
"""
author:WiGig11
usage
1.download .zip file
2.unzip
3.unzip .gz file
4.get file
"""
# Mapping of numeric labels to alphabetic labels (1 -> A, 2 -> B, ..., 26 -> Z)
label_mapping = {str(i + 1): letter for i, letter in enumerate(string.ascii_uppercase)}

# Function to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Read image file and save
def read_image(filename, path):
    with open(filename, "rb") as f:
        buf = f.read()

    index = 0
    magic, images, rows, columns = struct.unpack_from(">IIII", buf, index)
    index += struct.calcsize(">IIII")

    # Check and create directory for saving images
    ensure_dir(path)

    for i in tqdm(range(images), desc="Processing Images"):
        image = Image.new("L", (columns, rows))

        # Add progress bar for rows and columns processing
        for x in tqdm(range(rows), desc=f"Processing Row {i+1}/{images}", leave=False):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from(">B", buf, index)[0]))
                index += struct.calcsize(">B")

        # Flip and rotate image
        image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
        image2 = image1.rotate(90)
        image2.save(os.path.join(path, f"{i}.png"))

# Read label file and save as txt
def read_label(filename, save_filename):
    with open(filename, "rb") as f:
        buf = f.read()

    index = 0
    magic, labels = struct.unpack_from(">II", buf, index)
    index += struct.calcsize(">II")

    labelArr = []
    for x in tqdm(range(labels), desc="Processing Labels"):
        labelArr.append(int(struct.unpack_from(">B", buf, index)[0]))
        index += struct.calcsize(">B")

    # Ensure the directory for saving labels exists
    ensure_dir(os.path.dirname(save_filename))

    with open(save_filename, "w") as save:
        save.write(",".join(map(str, labelArr)))
        save.write("\n")
    
    print("Labels saved successfully.")

# Main program
if __name__ == "__main__":
    image_path = "gzip/emnist-letters-test-images-idx3-ubyte"
    label_path = "gzip/emnist-letters-test-labels-idx1-ubyte"
    label_save_path = "letters_test/label.txt"

    # Read dataset
    read_image(image_path, 'letters_test')
    # Read labels and parse them into a txt file
    read_label(label_path, label_save_path)

    # Read and save labels
    with open(label_save_path, "r") as labels_file:
        labels = labels_file.read().split(",")

    path = "letters_test/"
    output_dir = "letters_test/"

    for cnt in tqdm(range(124800), desc="Organizing Images"):
        image_path = os.path.join(path, f"{cnt}.png")
        img = cv2.imread(image_path)

        # Map numeric label to alphabetic label
        label = label_mapping.get(labels[cnt], labels[cnt])  # Default to original if no mapping found
        
        # Ensure each label directory exists
        label_dir = os.path.join(output_dir, label)
        ensure_dir(label_dir)

        cv2.imwrite(os.path.join(label_dir, f"{cnt}.png"), img)
