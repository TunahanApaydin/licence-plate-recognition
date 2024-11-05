import os
import re
import glob
   
def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])

    return parts

def get_sorted_files(file_dir):
    """
    Args:
        file_dir: Absolute path of file
    """
    list_subfolders_with_paths = [f.path for f in os.scandir(file_dir) if f.is_dir()]
    sorted_subfolders = sorted(list_subfolders_with_paths, key =  numerical_sort)

    return sorted_subfolders

def get_sorted_file_paths(file_dir, extensions):
    """
    Args:
        file_dir: Absolute path of file
        extensions: List of file extensions (e.g., ["jpg", "png"])
    """
    
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(file_dir, f'*.{ext}')))
        
    sorted_file_paths = sorted(paths, key = numerical_sort)
    sorted_names = [(os.path.basename(os.path.splitext(name)[0])) for name in sorted_file_paths]

    return sorted_names, sorted_file_paths
