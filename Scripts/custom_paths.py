import os
import shutil

def file_path(file_name):
    script_folder = os.path.dirname(os.path.abspath(__file__)) 
    data_folder = os.path.join(script_folder, "..", "Data") 
    file_path = os.path.join(data_folder,file_name) #File path
    return file_path

def move_files():
    script_folder = os.path.dirname(os.path.abspath(__file__)) 
    data_folder = os.path.join(script_folder, "..", "Data") 
    pkl_folder = os.path.join(script_folder, "..", "Models", "pkl")
    csv_folder = os.path.join(script_folder,"..", "Models", "csv")
    current_files = os.listdir() #Folder content
    file_prefix = "hall_of_fame_"
    matching_files = [file for file in current_files if file.startswith(file_prefix)]

    for file in matching_files:
        _, file_extension = os.path.splitext(file)

        if file_extension == ".pkl":
            destination_path = os.path.abspath(os.path.join(pkl_folder, file))
        elif file_extension == ".csv":
            destination_path = os.path.abspath(os.path.join(csv_folder, file))
        else:
            continue

        #print(f"Moving '{file}' to '{destination_path}'")
        
        try:
            shutil.move(file, destination_path)
        except FileNotFoundError as e:
            print(f"Error moving '{file}': {e}")

    for file in current_files:
        if file.endswith('bkup'):
            file_path = os.path.abspath(file)
            try:
                os.remove(file_path)
                #print(f"File '{file}' eliminated with success.")
            except OSError as e:
                print(f"Errore durante l'eliminazione di '{file}': {e}")