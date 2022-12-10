import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import params
from dataset import GazeFollow
import models

import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

basic_transformation = transforms.Compose([
    transforms.Resize((params.input_resolution, params.input_resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def print_debug(error, info):
    if error:
        print("\n[Error]: %s"%info)
    else:
        print("\n[Info]: %s"%info)

def pretrain_gazefollow():

    transform = basic_transformation
    # Prepare the gazefollow dataset
    print_debug(0, "Pretraining on gazefollow...")

    train_dataset = GazeFollow(params.gazefollow_train_imgs, params.gazefollow_train_label, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=params.gf_batch_size,
                                               shuffle=True,
                                               num_workers=0)
    
    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.GazeFollow_Model().to(device)
    heatmap_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.gf_lr)
    
    # Start training process
    total_train_batchs =  len(train_loader)

    for ep in range(params.gf_epochs):
        model.train(True)
        for batch, (imgs, faces, gt_heatmaps, _) in enumerate(train_loader):
            optimizer.zero_grad()
            gaze_preds = model(faces.cuda().to(device), imgs.cuda().to(device)).squeeze(1)
            loss = heatmap_loss(gaze_preds, gt_heatmaps)
            loss.backward()
            optimizer.step()
    
            if (batch%params.gf_train_printinfo==0):
                print_debug(0, "Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss:{:.4f}".format(ep, batch, total_train_batchs, loss))

    torch.save(model.state_dict(), '%s/%s'%(params.gf_weights_dir, 'epoch_%d.pt'%ep))

if __name__ == "__main__":
    random.seed(2022)
    pretrain_gazefollow()