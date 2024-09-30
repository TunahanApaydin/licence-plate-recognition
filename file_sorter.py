import os
import re
import glob

class FileSorter:
    def __init__(self):
        pass

    def numerical_sort(self, value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])

        return parts

    def get_sorted_files(self, file_dir):
        """
        Args:
            file_dir: Absolute path of file
        """
        list_subfolders_with_paths = [f.path for f in os.scandir(file_dir) if f.is_dir()]
        sorted_subfolders = sorted(list_subfolders_with_paths, key =  self.numerical_sort)

        return sorted_subfolders

    def get_sorted_names(self, file_dir, extention):
        """
        Args:
            file_dir: Absolute path of file
            extention: .xml - .txt - .jpg ...
        """
        paths = glob.glob(file_dir + extention)
        sorted_file_paths = sorted(paths, key = self.numerical_sort)
        sorted_names = [(os.path.basename(os.path.splitext(name)[0])) for name in sorted_file_paths]

        return sorted_names, sorted_file_paths
