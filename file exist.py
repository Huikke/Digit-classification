import os

file_path = "model.h5"

if os.path.exists(file_path):
    print("File exists.")
else:
    print("File does not exist.")