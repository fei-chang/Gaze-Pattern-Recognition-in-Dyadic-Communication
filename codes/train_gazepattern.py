import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
import os
import argparse


import config
import utils
from models import GazePattern_Model_V2, GazePattern_Model_Full
from dataset import DetectionDatasets, StaticGazes


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

basic_transformation = transforms.Compose([
    transforms.Resize((config.input_resolution, config.input_resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def run_detection(
    dataset_name,
    prefix,
    lr = config.detection_lr,
    batch_size = config.detection_batch_size,
    total_epochs = config.detection_epochs,
    save_epoch = config.detection_save_epoch,
    train_printinfo = config.detection_train_printinfo):

    # Prepare model
    
    model = GazePattern_Model_V2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare datasets
    if (dataset_name =='uco'):
        train_img_path = config.uco_imgs
        train_label = config.uco_train_label
    elif (dataset_name =='ava'):
        train_img_path = config.ava_imgs
        train_label = config.ava_train_label
    elif (dataset_name == 'oimg'):
        train_img_path = config.oimg_train_imgs
        train_label = config.oimg_train_label
    else:
        utils.print_debug(1, "Invalid dataset name")

    train_dataset = DetectionDatasets(train_img_path, train_label, basic_transformation, dataset_name)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            prefetch_factor=2,
                                            num_workers=8,
                                            sampler=DistributedSampler(train_dataset, shuffle=True))

    # Prepare Log and Save Info
    writer = SummaryWriter(config.detection_tensorboard_dir)
    weights_dir = config.detection_weights_dir
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    
    # Prepare Loss 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # Start training 
    total_train_batchs = len(train_loader)
    global_step = 0


    for ep in range(total_epochs):
        model.train(True)
        for batch, (imgs, faces, head_embeddings, labels, paths) in enumerate(train_loader):
            faces = faces.cuda().to(device)
            imgs = imgs.cuda().to(device)
            head_embeddings = head_embeddings.cuda().to(device)
            preds = model(faces, imgs, head_embeddings).squeeze()
            loss = criterion(preds, labels.float().cuda().to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (batch!=0)&((batch%train_printinfo==0)or(batch==total_train_batchs)):
                utils.print_debug(0, "Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss:{:.4f}".format(ep+1, batch, total_train_batchs, loss))
                writer.add_scalar('(%s)Loss'%prefix, loss.item(), global_step)
                global_step+=1

        if (ep!=0) and (ep%save_epoch==0):
            # Saving model weights
            torch.save(model.module.state_dict(), '%s/%s'%(weights_dir, '%s_epoch_%d.pt'%(prefix, ep)))

    writer.close()

def run_static_gazes(
    prefix = "gp_static",
    lr = config.staticgaze_lr,
    batch_size=config.staticgaze_batch_size,
    total_epochs = config.staticgaze_epochs,
    save_epoch = config.staticgaze_save_epoch,
    train_printinfo = config.staticgaze_train_printinfo,
    ddp=False):
    
    # Prepare model
    model = GazePattern_Model_Full()
    

    if ddp:
        dist.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        model = model.to(device)
        model = DDP(model)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
    # Prepare datasets
    train_dataset = StaticGazes(config.staticgaze_train_imgs, config.staticgaze_train_label, basic_transformation)
    if ddp:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            prefetch_factor=2,
                                            num_workers=8,
                                            sampler=DistributedSampler(train_dataset, shuffle=True))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            prefetch_factor=2,
                                            num_workers=8,
                                            shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(config.staticgaze_tensorboard_dir)

    weights_dir = config.staticgaze_weights_dir
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()

    # Start training process
    optimizer.zero_grad()
    total_train_batchs = len(train_loader)
    global_step = 0
    utils.print_debug(0, "Start Training...")
    for ep in range(total_epochs):
        model.train(True)
        for batch, (imgs, faces, head_embeddings, labels, paths) in enumerate(train_loader):
            faces = faces.cuda().to(device)
            imgs = imgs.cuda().to(device)
            head_embeddings = head_embeddings.cuda().to(device)
            preds = model(faces, imgs, head_embeddings).squeeze()
            loss = criterion(preds, labels.cuda().to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (batch!=0)&(batch%train_printinfo==0):
                utils.print_debug(0, "Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss:{:.4f}".format(ep, batch, total_train_batchs, loss))
                writer.add_scalar('(%s)Loss'%prefix, loss.item(), global_step)
                global_step+=1

        if ep%save_epoch==0:
            # Saving model weights
            if ddp:
                torch.save(model.module.state_dict(), '%s/%s'%(weights_dir, '%s_epoch_%d.pt'%(prefix, ep)))
            else:
                torch.save(model.state_dict(), '%s/%s'%(weights_dir, '%s_epoch_%d.pt'%(prefix, ep)))
    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str,  default='gp_static', help="oimg/ava/uco/gp_static")
    parser.add_argument("--prefix", type=str, default='gp_new', help="additional prefix for saving model.")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--print_every", type=int, default=200, help="print every ___ iterations")
    parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
    parser.add_argument("--ddp", type=bool, default=True)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    if args.dataset_name=='gp_static':
        run_static_gazes(args.prefix, args.lr, args.batch_size,
                         args.epochs, args.save_every, args.print_every, args.ddp)
    else:
        run_detection(args.dataset_name, args.prefix, args.lr,
                      args.batch_size, args.epochs, args.save_every, args.print_every)