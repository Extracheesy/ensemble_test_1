import pandas as pd
import numpy as np
import glob
import os

def merge_csv(extension):
    directory = "./raw_results/"
    current_directory = os.getcwd()
    print("current directory: ", current_directory)
    os.chdir(directory)
    print("current directory: ", os.getcwd())

    # all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    all_filenames = [i for i in glob.glob('*{}'.format(extension))]
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

    os.chdir(current_directory)
    print("current directory: ", os.getcwd())

    output_directory = "./computed_data/"
    os.chdir(output_directory)
    print("current directory: ", os.getcwd())

    # export to csv
    combined_csv.to_csv("combined_results.csv", index=False, encoding='utf-8-sig')

    return combined_csv