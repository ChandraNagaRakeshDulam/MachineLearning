import os
import pandas as pd
import numpy as np
import random
import albumentations
import argparse
import cv2

from imutils import paths
from tqdm import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-n', '--num-images', default=1000, type=int, help='total images to preprocess for every alphabet, space,nothing,delete')
args = vars(arg_parser.parse_args())

print(f"Preprocessing {args['num_images']} images from each alphabet.")

# getting the image data.
images_data = list(paths.list_images('./image_data/train_alphabets/train_alphabets'))
directory_data = os.listdir('./image_data/train_alphabets/train_alphabets')
directory_data.sort()

main_path = './image_data/train_alphabets/train_alphabets'

# getting 1000 images from the each alphabets, space, nothing, delete.
for idx, directory_dt  in tqdm(enumerate(directory_data), total=len(directory_data)):
    total_images = os.listdir(f"{main_path}/{directory_dt}")
    os.makedirs(f"./image_data/preprocessing_images/{directory_dt}", exist_ok=True)
    for i in range(args['num_images']): 

        # generating a random number from 0 to 2999 for the preprocessed images
        random_number = (random.randint(0, 2999))
        image = cv2.imread(f"{main_path}/{directory_dt}/{total_images[random_number]}")
        image = cv2.resize(image, (224, 224))

        cv2.imwrite(f"./image_data/preprocessing_images/{directory_dt}/{directory_dt}{i}.jpg", image)

print('preprocessing is done and files are saved')