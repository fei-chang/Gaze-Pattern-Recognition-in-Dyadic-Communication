import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, classification_report
from PIL import Image
import os
import cv2
import argparse

import config
import utils
from models import GazePattern_Model_V2, GazePattern_Model_Full
from dataset import DetectionDatasets, StaticGazes

import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

basic_transformation = transforms.Compose([
    transforms.Resize((config.input_resolution, config.input_resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def val_single_detection(model_weights, dataset_name):
    # Prepare model
    model = GazePattern_Model_V2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(model_weights)
    model.load_state_dict(weights)
    model.to(device)

    # Prepare datasets
    if (dataset_name =='uco'):
        test_img_path = config.uco_imgs
        test_label = config.uco_test_label
    elif (dataset_name =='ava'):
        test_img_path = config.ava_imgs
        test_label = config.ava_test_label
    elif (dataset_name == 'oimg'):
        test_img_path = config.oimg_test_imgs
        test_label = config.oimg_test_label
    else:
        utils.print_debug(1, 'Invalid dataset name')

    val_dataset = DetectionDatasets(test_img_path, test_label, basic_transformation, dataset_name, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=config.detection_batch_size,
                                            shuffle=False,
                                            prefetch_factor=2,
                                            num_workers=4)

    model.train(False)
    gt_ls = []; pred_ls = []
    with torch.no_grad():
        for val_batch, (val_img, val_face, val_head_embeddings, val_labels, _) in enumerate(val_loader):
            val_images = val_img.cuda().to(device)
            val_faces = val_face.cuda().to(device)
            val_head_embeddings = val_head_embeddings.cuda().to(device)
            val_preds = model(val_faces, val_images, val_head_embeddings)
            val_preds = val_preds.detach().cpu().squeeze()
            gt_ls.extend(val_labels)
            pred_ls.extend(val_preds)

        ap = average_precision_score(gt_ls, pred_ls)

        utils.print_debug(0, "AP:{:.4f}".format(ap))



def val_static_gaze(model_weights):
    
    # Prepare model
    model = GazePattern_Model_Full()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Prepare datasets

    val_dataset = StaticGazes(config.staticgaze_test_imgs, config.staticgaze_test_label, basic_transformation, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=config.staticgaze_batch_size,
                                            shuffle=False,
                                            prefetch_factor=2,
                                            num_workers=4)


    weights = torch.load(model_weights)
    model.load_state_dict(weights)
    model.to(device)

    # Start Evaluation

    model.train(False)
    gt_ls = []; pred_ls = []
    print("Running Evaluation...")
    with torch.no_grad():
        for val_batch, (val_img, val_face, val_head_embeddings, val_labels, _) in enumerate(val_loader):
            val_images = val_img.cuda().to(device)
            val_faces = val_face.cuda().to(device)
            val_head_embeddings = val_head_embeddings.cuda().to(device)
            output = model(val_faces, val_images, val_head_embeddings)
            output = output.detach().cpu().squeeze()
            val_preds = output.argmax(1)
            gt_ls.extend(val_labels)
            pred_ls.extend(val_preds)

        report = classification_report(gt_ls, pred_ls)
        acc_score = accuracy_score(gt_ls, pred_ls)
        utils.print_debug(0, "Acc:{:.4f}".format(acc_score))
        utils.print_debug(0, '\n%s'%report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="oimg/ava/uco/gp_static", default='gp_static')
    parser.add_argument("--model_weights", type=str, default="/home/changfei/GP_static/weights/gp_epoch_15.pt")

    args = parser.parse_args()
    if args.dataset_name=='gp_static':
        val_static_gaze(args.model_weights)
    else:
        val_single_detection(args.model_weights, args.dataset_name)
