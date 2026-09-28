from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "Models"


def encoder_loader_path():
    """
    Prompts the user in the console to select a subfolder, then a file within it.
    Returns the absolute path to the selected file as a string.
    """
    model_path = MODEL_DIR

    subfolders = [f for f in model_path.iterdir() if f.is_dir()]

    print("\n Folder Selection ")
    for index, folder in enumerate(subfolders):
        print(f"[{index + 1}] {folder.name}")

    # 3. Get user choice for the folder with error handling
    while True:
        try:
            folder_choice = int(input("\nEnter the number of the folder: ")) - 1
            if 0 <= folder_choice < len(subfolders):
                selected_folder = subfolders[folder_choice]
                break
            else:
                print("Invalid number. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    files = [f for f in selected_folder.iterdir() if f.is_file()]

    if not files:
        print(f"No files found inside '{selected_folder.name}'.")
        return None

    print(f"\n File Selection in '{selected_folder.name}' ")
    for index, file in enumerate(files):
        print(f"[{index + 1}] {file.name}")

    while True:
        try:
            file_choice = int(input("\nEnter the number of the file: ")) - 1
            if 0 <= file_choice < len(files):
                selected_file = files[file_choice]
                break
            else:
                print("Invalid number. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # 6. Return the full absolute path
    print(f"\n You selected: {selected_file.name}")
    return str(selected_file.resolve())


if __name__ == '__main__':
    encoder_loader_path()
