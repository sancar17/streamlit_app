import os

# Helper functions
def check_if_file_exists(filepath):
    return os.path.isfile(filepath)

def check_if_directory_exists(dirpath):
    return os.path.isdir(dirpath)