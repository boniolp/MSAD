import os
import re

folder_a = 'figures/final_pdf'
folder_b = '/home/sylli/Downloads/tmp'

for filename_b in os.listdir(folder_b):
    if filename_b.endswith('.pdf'):
        file_number_b = filename_b.split('-')[0]  # Assumes the number is before the first hyphen
        for filename_a in os.listdir(folder_a):
            if filename_a.endswith('.pdf') and filename_a.startswith(file_number_b):
                new_filename_b = filename_a
                os.rename(os.path.join(folder_b, filename_b), os.path.join(folder_b, new_filename_b))
                break
