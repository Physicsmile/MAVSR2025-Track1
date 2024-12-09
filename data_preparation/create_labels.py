import csv

input_csv = '../data/CAS-VSR-S101_zip/train/longform_transcripts.csv'
output_csv = '../data/CAS-VSR-S101/labels/CAS-VSR-S101_trans_train.csv'


with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile, delimiter='\t') 
    writer = csv.writer(outfile)

    for row in reader:
        path = row['path']
        duration = int(row['end_frame'])-int(row['start_frame'])+1 
        text = row['text']

        writer.writerow([f'CAS-VSR-S101', path, duration, text])

print(f"labels saved to {output_csv}")
