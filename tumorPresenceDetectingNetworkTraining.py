from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

load_dotenv()
noTumorDataset = os.getenv("datasetOfPicsWithNoTumor")
tumorDataset = os.getenv("datasetOfPicsWithTumor")

images_without_tumor= os.listdir(noTumorDataset)
images_with_tumor = os.listdir(tumorDataset)

dataset = []
label = []
input_image_size = 64


for i, image_name in enumerate(images_without_tumor):
    if(image_name.split('.')[1].lower() == 'jpg'):
        image_path = os.path.join(noTumorDataset, image_name)
        image = Image.open(image_path)
        # print(image)
        if image is not None:
            image = image.convert('RGB')
            image = image.resize((input_image_size, input_image_size))
            # print(image)
            dataset.append(np.array(image)) #converting img to pixel matrix
            label.append(0)  # 0 - no tumor, 1 - presence of tumor
        else:
            print(f"Failed to load image: {image_path}")
print(f"Pictures from noTumorDataset: {i}")

for i, image_name in enumerate(images_with_tumor):
    if(image_name.split('.')[1].lower() == 'jpg'):
        image_path = os.path.join(tumorDataset, image_name)
        image = Image.open(image_path)
        # print(image)
        if image is not None:
            image = image.convert('RGB')
            image = image.resize((input_image_size, input_image_size))
            # print(image)
            dataset.append(np.array(image)) #converting img to pixel matrix
            label.append(1)  # 0 - no tumor, 1 - presence of tumor
        else:
            print(f"Failed to load image: {image_path}")
print(f"Pictures from tumorDataset: {i}")

print(f"Images in dataset: {len(dataset)}")
print(f"Number of labels: {len(label)}")

dataset = np.array(dataset)
label = np.array(label)

dataset_train, dataset_test, label_train, label_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

print("Training dataset; images, sizeX, sizeY, channels :",dataset_train.shape)

dataset_train = normalize(dataset_train, axis=1)
dataset_test = normalize(dataset_test, axis=1)

label_train = to_categorical(label_train, num_classes=2)
label_test = to_categorical(label_test, num_classes=2)


model = Sequential()

kernel_size = 3
channels = 3
model.add(Conv2D(32, (kernel_size, kernel_size), input_shape=(input_image_size, input_image_size, channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (kernel_size, kernel_size), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (kernel_size, kernel_size), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))    #softmax will be better for multi-class classification, sigmoid for binary

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(dataset_train, label_train, batch_size=16, verbose=1, epochs=10,
          validation_data=(dataset_test, label_test), shuffle=False)

pathToNetwork = os.getenv("pathToPresenceDetectingNetwork")
model.save(pathToNetwork)