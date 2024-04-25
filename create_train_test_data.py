import shutil
import os


def create_train_test_data():
    """
    Requires the GTZAN dataset; download from kaggle and place in directory ./data/. https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    Makes a train/test split stratified by genre, and save the files in a new folder named /data_train_test/.

    Run using: python create_train_test_data.py
    """
    # Create directory for train and test data
    data_train_test_path = "data_train_test"
    train_path = os.path.join(data_train_test_path, "train")
    test_path = os.path.join(data_train_test_path, "test")

    # remove folder if it already exists
    if os.path.exists(data_train_test_path):
        shutil.rmtree(data_train_test_path)

    # Create the directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Create subdirectories inside train_path for each genre
    for genre_name in genre_names:
        genre_dir = os.path.join(train_path, genre_name)
        os.makedirs(genre_dir, exist_ok=True)

    # Create subdirectories inside test_path for each genre
    for genre_name in genre_names:
        genre_dir = os.path.join(test_path, genre_name)
        os.makedirs(genre_dir, exist_ok=True)

    print(f"Directory {data_train_test_path} created successfully.")

    path_to_genres = "./data/genres_original/"

    for genre_name in genre_names:
        # Construct full path to genre directory
        genre_dir = os.path.join(path_to_genres, genre_name)

        # Check if the directory exists
        if not os.path.isdir(genre_dir):
            print(f"Directory '{genre_dir}' does not exist.")
            continue

        # Loop over files in the genre directory
        for i, file_name in enumerate(os.listdir(genre_dir)):
            # Construct full path to file
            file_path = os.path.join(genre_dir, file_name)

            # Check if it's a file
            if os.path.isfile(file_path):
                # Determine destination directory based on condition
                if i % 5 == 0:  # every fifth file is going to test
                    destination = os.path.join(
                        os.path.join(test_path, genre_name), file_name
                    )
                else:  # otherwise to to train
                    destination = os.path.join(
                        os.path.join(train_path, genre_name), file_name
                    )

                # Copy the file to the appropriate destination
                shutil.copy(file_path, destination)

            else:
                print(f"'{file_path}' is not a file.")
    print(f"Succesfully created train/test data in folder {data_train_test_path}")


if __name__ == "__main__":
    create_train_test_data()
