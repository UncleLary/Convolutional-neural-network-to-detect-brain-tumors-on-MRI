from dotenv import load_dotenv
import os
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.src.utils import normalize
from serializeData import serialize_filenames, find_duplicates

load_dotenv()
tumorPredictionDataset = os.getenv("datasetOfPredictionPics")
# pathToNetwork = os.getenv(r"pathToPresenceDetectingNetwork")
pathToNetwork =  r"C:\Users\lakie\Desktop\Praca inżynierska\CNN_BrainMRI\Networks\tumorPresenceDetector10Epochs.keras"
noTumor_image = r"C:\Users\lakie\Desktop\Praca inżynierska\Data\DatasetForDetetctingPresenceOfTumorWithoutDuplicates\pred\pred0.jpg"
tumor_image = r"C:\Users\lakie\Desktop\Praca inżynierska\Data\DatasetForDetetctingPresenceOfTumorWithoutDuplicates\pred\pred2.jpg"

find_duplicates(tumorPredictionDataset)

prediction_images = os.listdir(tumorPredictionDataset)

model = load_model(pathToNetwork)

predictionDataset = []
label = []
input_image_size = 64

imageNoTum = Image.open(noTumor_image)
imageTum = Image.open(tumor_image)

imageNoTum = imageNoTum.convert('RGB')
imageNoTum = imageNoTum.resize((input_image_size, input_image_size))
imageNoTum = np.array(imageNoTum)
imageNoTum=np.expand_dims(imageNoTum, axis=0)

imageTum = imageTum.convert('RGB')
imageTum = imageTum.resize((input_image_size, input_image_size))
imageTum = np.array(imageTum)
imageTum=np.expand_dims(imageTum, axis=0)

for i, image_name in enumerate(prediction_images):
    if(image_name.split('.')[1].lower() == 'jpg'):
        image_path = os.path.join(tumorPredictionDataset, image_name)
        image = Image.open(image_path)
        if image is not None:
            image = image.convert('RGB')
            image = image.resize((input_image_size, input_image_size))
            image = np.array(image)
            np.expand_dims(image, axis=0)
            predictionDataset.append(image)
            # label.append(0)  # 0 - no tumor, 1 - presence of tumor
        else:
            print(f"Failed to load image: {image_path}")
print(f"Pictures from tumorPredictionDataset: {i}")

# predictionDataset = np.array(predictionDataset)
# predictionDataset = normalize(predictionDataset, axis=1)

for i, image_name in enumerate(predictionDataset):
    image_as_np_array = np.expand_dims(image_name, axis=0)
    result = model.predict(image_as_np_array)
    predicted_class = np.argmax(result)
    rounded_class = int(result[0][1])
    print(f"Ressult for prod{i} is: {result} | Estimated:{rounded_class}")



print("\n===================")
print("Reff test: ")

result = model.predict(imageNoTum)
predicted_class = np.argmax(result)

print(f"For No Tumor Img - pred0:{predicted_class}")
print("===================")

result = model.predict(imageTum)
predicted_class = np.argmax(result)

print(f"For Tumor img - pred2:{predicted_class}")