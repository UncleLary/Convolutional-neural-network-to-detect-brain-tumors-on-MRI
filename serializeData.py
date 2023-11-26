from PIL import Image
import imagehash
import os
import shutil

def serialize_filenames(folder_path,output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the folder
    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        try:
            image = Image.open(file_path)
        except:
            continue

        new_filename = f"image ({i + 1}).jpg"
        new_filepath = os.path.join(output_folder, new_filename)
        shutil.copy2(file_path, new_filepath)

    print(f"\nSerialization for {output_folder} is completed\n")


def find_duplicates(folder_path):

    hash_dict = {}

    duplicate_names = []

    print(f"Duplicates in {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            image = Image.open(file_path)
        except:
            continue

        image_hash = str(imagehash.dhash(image))

        # Check if the hash is already in the dictionary
        if image_hash in hash_dict:
            # Add the filename to the list of duplicates
            duplicate_names.append([filename] + hash_dict[image_hash])
            # Update the hash_dict to keep only the first filename in the group
            hash_dict[image_hash] = [hash_dict[image_hash][0]]

            print(", ".join([filename] + hash_dict[image_hash]))
        else:
            # Create a new entry in the dictionary
            hash_dict[image_hash] = [filename]
    print(f"\nSearching for duplicates in {folder_path} is finished.\n")
    if(len(duplicate_names) == 0):
        print("No duplicated images found.")
    else:
        return duplicate_names

def delete_duplicates(folder_path, duplicate_names):
    # Iterate through duplicate names and delete the corresponding files
    if(len(duplicate_names)!=0):
        for group in duplicate_names:
            for filename in group[1:]:
                file_path = os.path.join(folder_path, filename)

                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
                else:
                    print(f"File not found: {filename}. Skipped.")

        print(f"\nDeleting duplicates in {folder_path} is completed.")

    else:
        print(f"Nothing to delete in {folder_path}")