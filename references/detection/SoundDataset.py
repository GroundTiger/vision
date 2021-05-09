import os
import numpy as np
import pandas as pd
import argparse
import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as tf
from torchvision.io import read_image
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()


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

class SoundDataset(object):
    def __init__(self, csv, transforms):
        self.transforms = transforms
        colName = ["filepath", "label_0", "time_0", "label_1", "time_1", "label_2", "time_2"]
        self.df = pd.read_csv(csv, names=colName)

    def __getitem__(self, idx):
        img_path = self.df["filepath"][idx]
        img = Image.open(img_path).convert("RGB")
        time_start = 0
        boxes = []
        labels = []
        for i in range(3):
            label = "label_" + str(i)
            time = "time_" + str(i)
            if not pd.isnull(self.df[label][idx]) and not pd.isnull(self.df[time][idx]):
                labels.append(label_map[self.df[label][idx]])
                time_end = time_start + self.df[time][idx]
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.df.shape[0]


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    parser = argparse.ArgumentParser(description = "train model according to given csv file")
    parser.add_argument("input", help = "input csv file")
    args = parser.parse_args()

    csv = args.input
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = len(label_map)+1
    # use our dataset and defined transformations
    dataset = SoundDataset(csv, get_transform(train=True))
    dataset_test = SoundDataset(csv, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-200])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-200:])
    dataset_print = torch.utils.data.Subset(dataset, indices[:10])
    print(dataset_print)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model, 'D:/Qiang/AI/pytorch/object_detection/models/models.pt')
    print("That's it!")

def inference():
    parser = argparse.ArgumentParser(description = "inference")
    parser.add_argument("imgN", help = "numbering of image")
    args = parser.parse_args()
    num = args.imgN
    path = 'D:/Acoustic Result/Event Tagging/1000_2'
    imgName = '_EventTagging_' + num.zfill(5) + '_STFT4096__ACC.jpg'
    filename = os.path.join(path, imgName)
    img = Image.open(filename).convert("RGB")
    img = tf.ToTensor()(img)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    path = 'D:/Acoustic Result/Event Tagging/model_100/model_49.pth'
    model = torch.load(path)
    model.eval()
    model.to(device)
    img = img.to(device)
    outputs = model(img.unsqueeze(0))
    print(outputs)

    colors = ['red', 'blue', 'green', 'black']
    boxes = outputs[0]['boxes']
    labels = [label_map_inv[int(i)] for i in outputs[0]['labels'].cpu()]
    print(labels)
    N = 4
    img = tf.ConvertImageDtype(dtype=torch.uint8) (img.cpu())
    result = draw_bounding_boxes(img, boxes=boxes[:N], labels=labels[:N], colors=colors[:N], width=3, fill=False, font_size=40)
    show(result)

if __name__ == "__main__":
    #main()
    inference()
