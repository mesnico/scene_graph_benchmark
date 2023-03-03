# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os
import os.path as op
import json
import cv2
import base64
from shutil import copyfile
import argparse
import tqdm

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
from maskrcnn_benchmark.structures.tsv_file_ops import generate_linelist_file
from maskrcnn_benchmark.structures.tsv_file_ops import generate_hw_file
from maskrcnn_benchmark.structures.tsv_file import TSVFile
from maskrcnn_benchmark.data.datasets.utils.image_ops import img_from_base64

# To generate a tsv file:
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default=None,
                        help='Destination folder (created if not existing)')
parser.add_argument('--image_path', default=None,
                        help='Path to image folder')
opt = parser.parse_args()
print(opt)

dataset_path = opt.dataset_path #"/home/nicola/scene_graph_test_dataset"
data_path = opt.image_path # dataset_path+"/images/"
img_list = os.listdir(data_path)
tsv_file = dataset_path+"/train.tsv"
label_file = dataset_path+"/train.label.tsv"
hw_file = dataset_path+"/train.hw.tsv"
linelist_file = dataset_path+"/train.linelist.tsv"

if not op.exists(dataset_path):
   os.makedirs(dataset_path)
copyfile("tools/mini_tsv/data/train.yaml", dataset_path+"/train.yaml")
copyfile("tools/mini_tsv/data/VG-SGG-dicts-vgoi6-clipped.json", dataset_path+"/VG-SGG-dicts-vgoi6-clipped.json")

rows = []
rows_label = []
rows_hw = []
for img_p in tqdm.tqdm(img_list):
    img_key = img_p.split('.')[0]
    img_path = op.join(data_path, img_p)
    img = cv2.imread(img_path)
    img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

    # Here is just a toy example of labels.
    # The real labels can be generated from the annotation files
    # given by each dataset. The label is a list of dictionary 
    # where each box with at least "rect" (xyxy mode) and "class"
    # fields. It can have any other fields given by the dataset.
    labels = []
    labels.append({"rect": [1, 1, 30, 40], "class": "Dog"})
    labels.append({"rect": [2, 3, 100, 100], "class": "Cat"})

    row = [img_key, img_encoded_str]
    rows.append(row)

    row_label = [img_key, json.dumps(labels)]
    rows_label.append(row_label)

    height = img.shape[0]
    width = img.shape[1]
    row_hw = [img_key, json.dumps([{"height":height, "width":width}])]
    rows_hw.append(row_hw)

tsv_writer(rows, tsv_file)
tsv_writer(rows_label, label_file)
tsv_writer(rows_hw, hw_file)

# generate linelist file
generate_linelist_file(label_file, save_file=linelist_file)

'''
# To access a tsv file:
# 1) Use tsv_reader to read dataset in given order
rows = tsv_reader("tools/mini_tsv/data/train.tsv")
rows_label = tsv_reader("tools/mini_tsv/data/train.label.tsv")
for row, row_label in zip(rows, rows_label):
    img_key = row[0]
    labels = json.loads(row_label[1])
    img = img_from_base64(row[1])

# 2) use TSVFile to access dataset at any given row.
tsv = TSVFile("tools/mini_tsv/data/train.tsv")
row = tsv.seek(1) # to access the second row 
img_key = row[0]
img = img_from_base64(row[1])
'''


