import os
import sys
import glob
import pandas as pd
import torch
from PIL import Image

label_map = { 'Triangle' : 1,
              'Sawtooth' : 2,
              'Square' : 3,
              'Sine' : 4}

label_map_inv = { 1 : 'Triangle',
                  2 : 'Sawtooth',
                  3 : 'Square',
                  4 : 'Sine'}

def convert_time_pixel(start, end):
    xOrigin = 83.0
    ymin = 37
    ymax = 451
    timeVal = 6.5
    length = 1137
    nPixelPerSec = (length-xOrigin)/timeVal
    xmin = int(start * nPixelPerSec + xOrigin)
    xmax = int(end * nPixelPerSec + xOrigin)
    return [xmin, ymin, xmax, ymax]

class EventTag(object):
    def __init__(self, csv, transforms):
        self._transforms = transforms
        colName = ["filepath", "label_0", "time_0", "label_1", "time_1", "label_2", "time_2"]
        self._df = pd.read_csv(csv, names=colName)

    def __getitem__(self, idx):
        img_path = self._df["filepath"][idx]
        img = Image.open(img_path).convert("RGB")
        time_start = 0
        boxes = []
        labels = []
        for i in range(3):
            label = "label_" + str(i)
            time = "time_" + str(i)
            if not pd.isnull(self._df[label][idx]) and not pd.isnull(self._df[time][idx]):
                labels.append(label_map[self._df[label][idx]])
                time_end = time_start + self._df[time][idx]
                time_end = min(time_end, 6.5)
                box = convert_time_pixel(time_start, time_end)
                boxes.append(box)
                time_start = time_end
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def __len__(self):
        return self._df.shape[0]

def get_event(root, transforms, mode='instances'):
    pattern = os.path.join(root, 'Tag*.csv')
    csvFiles = glob.glob(pattern)
    if len(csvFiles) != 1:
        print("Error: find more than 1 csv file in {}".format(root))
        sys.exit()
    dataset = EventTag(csvFiles[0], transforms=transforms)
    return dataset
