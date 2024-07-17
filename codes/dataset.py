import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF


import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import os
import glob
import warnings

import utils
import config
warnings.simplefilter(action='ignore', category=FutureWarning)

# GazeFollow Dataset
class GazeFollow(Dataset):
    """
    Dataset for pretraining on gazefollow-extended dataset.
    Some parts of the code is adopted from https://github.com/ejcgt/attention-target-detection
    """

    def __init__(self, data_dir, csv_path, transform, test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.test = test

        self.input_size = config.input_resolution
        self.output_size = config.output_resolution

        if test:
            df = pd.read_csv(csv_path)
            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max']].groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)

        else:
            df = pd.read_csv(csv_path)
            df.reset_index(inplace=True)
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                               'gaze_y', 'inout']]
            self.X_train = df['path']
            self.length = len(df)


    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for _, row in g.iterrows():
                path = row['path']
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
        else:
            path = self.X_train.iloc[index]
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout = self.y_train.iloc[index]

        # expand face bbox a bit
        k = config.expanding_face_factor
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        if self.test:
            imsize = torch.IntTensor([width, height])
        else:
            ## data augmentation
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y

                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)

                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        # Crop the face
        head_img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        if self.transform is not None:
            img = self.transform(img)
            head_img = self.transform(head_img)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if self.test:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap = utils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         3,
                                                         type='Gaussian')
            gaze_heatmap /= num_valid
        else:
            gaze_heatmap = utils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')

        head_embeddings = utils.get_head_embeddings(np.array([x_min/width, y_min/height, x_max/width, y_max/height]))
        full_path = os.path.join(self.data_dir, path)   
        if self.test:
            return img, head_img, head_embeddings, gaze_heatmap, cont_gaze, imsize, full_path
        else:
            return img, head_img, head_embeddings, gaze_heatmap, full_path

    def __len__(self):
        return self.length


# LAEO_UCO/AVA Dataset
class DetectionDatasets(Dataset):
    """
    Dataset for datasets of LAEO gaze pattern detection task: Including LAEO-UCO/LAEO-AVA/OIMG
    """

    def __init__(self, data_dir, csv_path, transform, dataset_name, test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.test = test
        self.dataset_name = dataset_name

        self.input_size = config.input_resolution
        self.output_size = config.output_resolution
        
        self.df = pd.read_csv(csv_path)
        self.length = len(self.df)

    def __getitem__(self, index):
        
        if (self.dataset_name in ['uco', 'ava']):
            vid_id, frame_num, p1_xmin, p1_ymin, p1_xmax, p1_ymax, p2_xmin, p2_ymin, p2_xmax, p2_ymax, label = self.df.iloc[index]
            img_path = os.path.join(self.data_dir, '%s/%06d.jpg'%(vid_id, frame_num))
        
        elif self.dataset_name == 'oimg':
            img_id, p1_xmin, p1_ymin, p1_xmax, p1_ymax, p2_xmin, p2_ymin, p2_xmax, p2_ymax, label = self.df.iloc[index]
            img_path = os.path.join(self.data_dir, '%s.jpg'%img_id)


        img = Image.open(img_path)
        img = img.convert('RGB')
        width, height = img.size
        if self.dataset_name == 'oimg': #（OIMG data annotations save head boxes in 0 - 1 scale)
            p1_xmin, p1_xmax, p2_xmin, p2_xmax = np.array([p1_xmin, p1_xmax, p2_xmin, p2_xmax])*width
            p1_ymin, p1_ymax, p2_ymin, p2_ymax = np.array([p1_ymin, p1_ymax, p2_ymin, p2_ymax])*height 
        
        p1_xmin, p1_ymin, p1_xmax, p1_ymax = map(float, [p1_xmin, p1_ymin, p1_xmax, p1_ymax])
        p2_xmin, p2_ymin, p2_xmax, p2_ymax = map(float, [p2_xmin, p2_ymin, p2_xmax, p2_ymax])
        
        if not self.test:
            ## data augmentation
            # Jitter (expansion-only) bounding box size(s)
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                random_person = np.random.randint(3)
                if (random_person!=0):
                    p1_xmin -= k * abs(p1_xmax - p1_xmin)
                    p1_ymin -= k * abs(p1_ymax - p1_ymin)
                    p1_xmax += k * abs(p1_xmax - p1_xmin)
                    p1_ymax += k * abs(p1_ymax - p1_ymin)
                if (random_person!=1):
                    p2_xmin -= k * abs(p2_xmax - p2_xmin)
                    p2_ymin -= k * abs(p2_ymax - p2_ymin)
                    p2_xmax += k * abs(p2_xmax - p2_xmin)
                    p2_ymax += k * abs(p2_ymax - p2_ymin)
            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the faces.
                crop_x_min = np.min([p1_xmin, p2_xmin])
                crop_y_min = np.min([p1_ymin, p2_ymin])
                crop_x_max = np.max([p1_xmax, p2_xmax])
                crop_y_max = np.max([p1_ymax, p2_ymax])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                p1_xmin, p1_ymin, p1_xmax, p1_ymax = p1_xmin - offset_x, p1_ymin - offset_y, p1_xmax - offset_x, p1_ymax - offset_y
                p2_xmin, p2_ymin, p2_xmax, p2_ymax = p2_xmin - offset_x, p2_ymin - offset_y, p2_xmax - offset_x, p2_ymax - offset_y
                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                p1_x_max_2 = width - p1_xmin
                p1_x_min_2 = width - p1_xmax
                p1_xmax = p1_x_max_2
                p1_xmin = p1_x_min_2

                p2_x_max_2 = width - p2_xmin
                p2_x_min_2 = width - p2_xmax
                p2_xmax = p2_x_max_2
                p2_xmin = p2_x_min_2
                
            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        # Crop the face
        p1_hbox = np.array([p1_xmin, p1_ymin, p1_xmax, p1_ymax]).astype(int)
        p2_hbox = np.array([p2_xmin, p2_ymin, p2_xmax, p2_ymax]).astype(int)
        head_img1 = img.crop(p1_hbox)
        head_img2 = img.crop(p2_hbox)
        
        if self.transform is not None:
            img = self.transform(img)
            head_img1 = self.transform(head_img1)
            head_img2 = self.transform(head_img2)

        head_imgs = torch.concat([head_img1, head_img2])
        
        head_embeddings1 = utils.get_head_embeddings(np.array([p1_xmin/width, p1_ymin/height, p1_xmax/width, p1_ymax/height]))
        head_embeddings2 = utils.get_head_embeddings(np.array([p2_xmin/width, p2_ymin/height, p2_xmax/width, p2_ymax/height]))
        head_embeddings = torch.concat([head_embeddings1, head_embeddings2], dim = -1)
        

        return img, head_imgs, head_embeddings, label, img_path

    def __len__(self):
        return self.length

# Our Dataset
class StaticGazes(Dataset):
    def __init__(self, data_dir, csv_path, transform, test=False):
        df = pd.read_csv(csv_path)
        self.df = df
        self.length = len(df)
        self.transform = transform
        self.data_dir = data_dir
        self.test = test
        self.patterns = {
            'Share':0,
            'Mutual':1,
            'Single':2,
            'Miss':3,
            'Void':4
        }
    
    def __getitem__(self, index):
        vid_id, frame, p1_xmin, p1_ymin, p1_xmax, p1_ymax, p2_xmin, p2_ymin, p2_xmax, p2_ymax, label = self.df.iloc[index]
        path = '%s/%d/%05d.jpg'%(self.data_dir, vid_id, frame)
        img = Image.open(path)
        img = img.convert('RGB')
        width, height = img.size 
        p1_xmin, p1_ymin, p1_xmax, p1_ymax = map(float, [p1_xmin, p1_ymin, p1_xmax, p1_ymax])
        p2_xmin, p2_ymin, p2_xmax, p2_ymax = map(float, [p2_xmin, p2_ymin, p2_xmax, p2_ymax])
        
        if not self.test:
            ## data augmentation
            # Jitter (expansion-only) bounding box size(s)
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                random_person = np.random.randint(3)
                if (random_person!=0):
                    p1_xmin -= k * abs(p1_xmax - p1_xmin)
                    p1_ymin -= k * abs(p1_ymax - p1_ymin)
                    p1_xmax += k * abs(p1_xmax - p1_xmin)
                    p1_ymax += k * abs(p1_ymax - p1_ymin)
                if (random_person!=1):
                    p2_xmin -= k * abs(p2_xmax - p2_xmin)
                    p2_ymin -= k * abs(p2_ymax - p2_ymin)
                    p2_xmax += k * abs(p2_xmax - p2_xmin)
                    p2_ymax += k * abs(p2_ymax - p2_ymin)
            # Random Crop
            if np.random.random_sample() <= 0.3:
                # Calculate the minimum valid range of the crop that doesn't exclude the faces.
                crop_x_min = np.min([p1_xmin, p2_xmin])
                crop_y_min = np.min([p1_ymin, p2_ymin])
                crop_x_max = np.max([p1_xmax, p2_xmax])
                crop_y_max = np.max([p1_ymax, p2_ymax])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                p1_xmin, p1_ymin, p1_xmax, p1_ymax = p1_xmin - offset_x, p1_ymin - offset_y, p1_xmax - offset_x, p1_ymax - offset_y
                p2_xmin, p2_ymin, p2_xmax, p2_ymax = p2_xmin - offset_x, p2_ymin - offset_y, p2_xmax - offset_x, p2_ymax - offset_y
                width, height = crop_width, crop_height


            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                p1_x_max_2 = width - p1_xmin
                p1_x_min_2 = width - p1_xmax
                p1_xmax = p1_x_max_2
                p1_xmin = p1_x_min_2

                p2_x_max_2 = width - p2_xmin
                p2_x_min_2 = width - p2_xmax
                p2_xmax = p2_x_max_2
                p2_xmin = p2_x_min_2
                
            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

            # Random SSJ
            if np.random.random_sample() <= 0.3:
                min_lsj_scale = 0.7
                scale = min_lsj_scale + np.random.random_sample()*(1-min_lsj_scale)
                nh, nw = int(height * scale), int(width * scale)
                img = TF.resize(img, (nh, nw))
                img = ImageOps.expand(img, (0, 0, int(width - nw), int(height - nh)))
                p1_xmin, p1_ymin, p1_xmax, p1_ymax = map(int,[p1_xmin*scale, p1_ymin*scale, p1_xmax*scale, p1_ymax*scale])
                p2_xmin, p2_ymin, p2_xmax, p2_ymax = map(int,[p2_xmin*scale, p2_ymin*scale, p2_xmax*scale, p2_ymax*scale])

        # Crop the face
        p1_hbox = np.array([p1_xmin, p1_ymin, p1_xmax, p1_ymax]).astype(int)
        p2_hbox = np.array([p2_xmin, p2_ymin, p2_xmax, p2_ymax]).astype(int)
        head_img1 = img.crop(p1_hbox)
        head_img2 = img.crop(p2_hbox)

        if self.transform is not None:
            img = self.transform(img)
            head_img1 = self.transform(head_img1)
            head_img2 = self.transform(head_img2)
        head_imgs = torch.concat([head_img1, head_img2])   
        head_embeddings1 = utils.get_head_embeddings(np.array([p1_xmin/width, p1_ymin/height, p1_xmax/width, p1_ymax/height]))
        head_embeddings2 = utils.get_head_embeddings(np.array([p2_xmin/width, p2_ymin/height, p2_xmax/width, p2_ymax/height]))
        head_embeddings = torch.concat([head_embeddings1, head_embeddings2], dim = -1)
        
        label = self.patterns[label]
        
        return img, head_imgs, head_embeddings, label, path

    def __len__(self):
        return self.length