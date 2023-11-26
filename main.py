from dotenv import load_dotenv
import os
import numpy as np
import tensorflow


from serializeData import serialize_filenames, find_duplicates, delete_duplicates

load_dotenv()
tumorDataset = os.getenv("datasetOfPicsWithTumor")
noTumorDataset = os.getenv(r"datasetOfPicsWithNoTumor")
tumorPredictionDataset = os.getenv("datasetOfPredictionPics")

healthy_brain_training = os.getenv("healthyBrainTraining")
glioma_tumor_training = os.getenv("GliomaTumorTraining")
meningioma_tumor_training = os.getenv("MeningiomaTumorTraining")
pituitary_tumor_training = os.getenv("PituitaryTumorTraining")

healthy_brain_testing = os.getenv("healthyBrainTesting")
glioma_tumor_testing = os.getenv("GliomaTumorTesting")
meningioma_tumor_testing = os.getenv("MeningiomaTumorTesting")
pituitary_tumor_testing = os.getenv("PituitaryTumorTesting")

serialized_healthy_brain_training = os.getenv("serializedHealthyBrainTraining")
serialized_glioma_tumor_training = os.getenv("serializedGliomaTumorTraining")
serialized_meningioma_tumor_training = os.getenv("serializedMeningiomaTumorTraining")
serialized_pituitary_tumor_training = os.getenv("serializedPituitaryTumorTraining")

serialized_healthy_brain_testing = os.getenv("serializedHealthyBrainTesting")
serialized_glioma_tumor_testing = os.getenv("serializedGliomaTumorTesting")
serialized_meningioma_tumor_testing = os.getenv("serializedMeningiomaTumorTesting")
serialized_pituitary_tumor_testing = os.getenv("serializedPituitaryTumorTesting")

if __name__ == '__main__':

    folder_path = tumorPredictionDataset
    # new_folder_path = normalized_pituitary_tumor_testing

    # print(f'From folder: {folder_path}  to: {new_folder_path}')
    # serialize_filenames(folder_path, new_folder_path)

    # duplicates = find_duplicates(folder_path)
    #
    # delete_duplicates(folder_path, duplicates)


