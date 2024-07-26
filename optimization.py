
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
class preprocess_data:
    def visualization_images(selfself,dir_path,nimages):
        fig, axs = plt.subplots(2,2,figsize = (10, 10))
        dpath=dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img = os.path.join(dpath, i, train_class[j])
                img = cv2.imread(img)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count += 1
        fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath=dir_path
        #count the number of images in the dataset
        train=[]
        labels=[]
        for i in os.listdir(dpath):
            #get the list of images in a given class
            train_class=os.listdir(os.path.join(dpath,i))
            for j in train_class:
                img=os.path.join(dpath,i,j)
                train.append(img)
                labels.append(i)
        print("number of images:{}\n".format(len(train)))
        print("number of image labels:{}\n".format(len(labels)))
        retina_df=pd.DataFrame({'Image':train, 'Labels':labels})
        print(retina_df)
        return retina_df,train,labels

    def generate_train_test_images(self,image_df,train,label):
        train, test= train_test_split(image_df,test_size=0.2)
        print(test)
        train_datagen = ImageDataGenerator(rescale=1. / 225,shear_range=0.2, validation_split=0.15)
        test_datagen = ImageDataGenerator(rescale=1. / 225)
        train_generator = train_datagen.flow_from_dataframe(
            train,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28,28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=40,
            subset='training'
        )
        validation_generator = train_datagen.flow_from_dataframe(
            train,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=40,
            subset='validation'
        )
        test_generator = test_datagen.flow_from_dataframe(
            test,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=40,
        )
        print(f"Train images shape:{train.shape}")
        print(f"testing images shape: {test.shape}")
        return train_generator, test_generator, validation_generator


