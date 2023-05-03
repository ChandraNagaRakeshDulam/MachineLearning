import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths

# getting the preprocessed images
images_data = list(paths.list_images('./image_data/preprocessing_images'))

# creating a dataframe
sign_data = pd.DataFrame()

image_names = []
for i, image_path in tqdm(enumerate(images_data), total=len(images_data)):
    image_name = image_path.split(os.path.sep)[-2]
    sign_data.loc[i, 'image_path'] = image_path

    image_names.append(image_name)

image_names = np.array(image_names)
# onehot encoding the data (which takes categorical data as input and output as numpy array)
data_lb = LabelBinarizer()
image_names = data_lb.fit_transform(image_names)

print(f"The onehot encoded image_names: {image_names[0]}")
print(f"Mapping the onehot encoded names to alphabets: {data_lb.classes_[0]}")
print(f"Total images: {len(image_names)}")

for i in range(len(image_names)):
    index = np.argmax(image_names[i])
    sign_data.loc[i, 'target'] = int(index)

sign_data = sign_data.sample(frac=1).reset_index(drop=True)

sign_data.to_csv('./image_data/csv_data.csv', index=False)


joblib.dump(data_lb, './predictions/data_lb.pkl')
print('data_lb pickled file is saved')
print(sign_data.head(10))