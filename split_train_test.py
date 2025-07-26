import os
import csv
import random
from sklearn.model_selection import train_test_split


makeup_dir = r"E:\datasets\MT\all\images\makeup"
no_makeup_dir =  r"E:\datasets\MT\all\images\non-makeup"

split_dir = './splits'
os.makedirs(split_dir, exist_ok=True)

no_makeup_files = sorted(os.listdir(no_makeup_dir))
makeup_files = sorted(os.listdir(makeup_dir))

no_makeup_train, no_makeup_test = train_test_split(no_makeup_files, test_size=0.2, random_state=42)
makeup_train, makeup_test = train_test_split(makeup_files, test_size=0.2, random_state=42)

def save_csv(file_list, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename'])
        for file in file_list:
            writer.writerow([file])

# Save CSVs
save_csv(no_makeup_train, os.path.join(split_dir, 'no_makeup_train.csv'))
save_csv(no_makeup_test, os.path.join(split_dir, 'no_makeup_test.csv'))
save_csv(makeup_train, os.path.join(split_dir, 'makeup_train.csv'))
save_csv(makeup_test, os.path.join(split_dir, 'makeup_test.csv'))

print(f"Saved splits in: {split_dir}")
