import csv
import os
import re

csv_file_path = "../../../data/CAS-VSR-S101/labels/CAS-VSR-S101_trans_val.csv"

output_file_path = 'val_ground_truth_sorted_by_sample_id.txt'

    
with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    with open(output_file_path, 'w', encoding='utf-8') as txtfile:
        for row in csv_reader:
            _,d,frm_len,content = row
            cleaned_text = re.sub(r'[^\w\u4e00-\u9fff]', '', content)
            if int(frm_len)<=1000:
                txtfile.write(cleaned_text + '\n')

