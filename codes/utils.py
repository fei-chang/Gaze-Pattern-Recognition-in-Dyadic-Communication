import torch
import numpy as np
import random
import os

def to_numpy_img(tensor):
    np_arr = to_numpy(tensor)
    img = np.transpose(np_arr, (1,2,0))
    return img
    
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img = img/np.max(img) # normalize heatmap so it has max value of 1
    return to_torch(img)


def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int,[p[0]*w.float(), p[1]*h.float()])
            x = min(x, w-1)
            y = min(y, h-1)
            target_map[y, x] = 1
    return target_map

def get_center_radius(x_min, y_min, x_max, y_max, img_width, img_height):
    #returnt the position of head center and head radius in scale 0-1
    head_x = (x_max+x_min)/(img_width*2)
    head_y = (y_max+y_min)/(img_height*2)
    head_center = torch.Tensor([[max(min(head_x, 1), 0), max(min(head_y, 1), 0)]])
    radius = max(x_max-x_min, y_max-y_min)/(img_width+img_height)

    return head_center, radius

def get_head_embeddings(head_box):
    """
    input: head_box: (xmin, ymin, xmax, ymax) of range 0 to 1
    generate a heatmap of size 32 x 32 where the pixel value equal to 1 
    if a head is found in that grid and 0 otherwise, and reshape it to a tensor of size 1x1024
    """
    xmin, ymin, xmax, ymax = np.array(head_box*32).astype(int)
    
    head_embeddings = np.zeros((32, 32))
    head_embeddings[ymin:ymax, xmin:xmax] = 1
    head_embeddings = to_torch(head_embeddings).view(1, 1024).type(torch.float)
    
    return head_embeddings 

def print_debug(error, info):
    if error:
        print("[Error]: %s"%info)
    else:
        print("[Info]: %s"%info)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
