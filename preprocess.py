preprocess.py:

import os
import pandas as pd import cv2
from matplotlib import pyplot as plt


class preprocess_data:
# write method to visualize images
def visualize_images(self, dir_path, nimages): fig, axs = plt.subplots(2, 2, figsize=(10, 10)) dpath = dir_path
count = 0
for i in os.listdir(dpath):
# get the list of images in a given class train_class = os.listdir(os.path.join(dpath, i)) # plot the images
for j in range(nimages):
img = os.path.join(dpath, i, train_class[j]) img = cv2.imread(img) axs[count][j].title.set_text(i) axs[count][j].imshow(img)
count += 1 fig.tight_layout() plt.show(block=True)


# write method to preprocess the data def preprocess(self, dir_path):
 
dpath = dir_path
# count the number of images in the dataset train = []
labels = []
for i in os.listdir(dpath):
# get the list of images in a given class train_class = os.listdir(os.path.join(dpath, i)) for j in train_class:
img = os.path.join(dpath, i, j) train.append(img) labels.append(i)
print("number of images:{}\n".format(len(train))) print("number of image labels:{}\n".format(len(labels))) retina_df = pd.DataFrame({'Image': train, 'Labels': labels}) print(retina_df)
return retina_df, train, labels
