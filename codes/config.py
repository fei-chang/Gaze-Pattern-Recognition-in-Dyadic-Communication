input_resolution = 224
output_resolution = 64
expanding_face_factor = 0.1

# Dataset information
# GazeFollow
gazefollow_imgs= "/home/changfei/X_Dataset/D_gazefollow_extended"
gazefollow_train_label = "/home/changfei/X_Dataset/D_gazefollow_extended/train_annotations_inuse.csv"
gazefollow_val_label = "/home/changfei/X_Dataset/D_gazefollow_extended/test_annotations_inuse.csv"

# LAEO-UCO
uco_imgs = '/home/changfei/X_Dataset/D_LAEO-UCO/ucolaeodb/frames'
uco_train_label = '/home/changfei/X_Dataset/D_LAEO-UCO/ucolaeodb/annotations/pair_train_annotations.csv'
uco_test_label = '/home/changfei/X_Dataset/D_LAEO-UCO/ucolaeodb/annotations/pair_test_annotations.csv'

# LAEO-AVA
ava_imgs = '/home/changfei/X_Dataset/D_LAEO-AVA/frames'
ava_train_label = '/home/changfei/X_Dataset/D_LAEO-AVA/annotations_csv/valid_train_annotations.csv'
ava_test_label = '/home/changfei/X_Dataset/D_LAEO-AVA/annotations_csv/valid_val_annotations.csv'

# LAEO-OIMG
oimg_train_imgs = '/home/changfei/X_Dataset/D_LAEO-OIMG/train'
oimg_test_imgs = '/home/changfei/X_Dataset/D_LAEO-OIMG/test'
oimg_train_label = '/home/changfei/X_Dataset/D_LAEO-OIMG/valid_train_annotations.csv'
oimg_test_label = '/home/changfei/X_Dataset/D_LAEO-OIMG/valid_test_annotations.csv'

# StaticGazes
staticgaze_train_imgs = '/home/changfei/X_Dataset/D_StaticGazes/frames/train'
staticgaze_test_imgs = '/home/changfei/X_Dataset/D_StaticGazes/frames/test'
staticgaze_train_label = '/home/changfei/X_Dataset/D_StaticGazes/annotations/train_annotations.csv'
staticgaze_test_label = '/home/changfei/X_Dataset/D_StaticGazes/annotations/test_annotations.csv'

# model training parameters
# GazeFollow Pretrain
gf_lr = 5e-4
gf_batch_size = 128
gf_epochs = 5
gf_save_epoch = 1
gf_train_printinfo = 200
gf_tensorboard_dir = '/home/changfei/GP_static/tensorboard/gf_pretrain'
gf_weights_dir = '/home/changfei/GP_static/weights/gf_pretrain'

# Mutual Detection Task
detection_lr = 5e-4
detection_batch_size = 128
detection_epochs = 10
detection_save_epoch = 2
detection_train_printinfo = 200
detection_tensorboard_dir = '/home/changfei/GP_static/tensorboard/detection'
detection_weights_dir = '/home/changfei/GP_static/weights/detection'

# Static Gaze Classification
staticgaze_lr = 2e-4
staticgaze_batch_size = 128
staticgaze_epochs = 30
staticgaze_save_epoch = 1
staticgaze_train_printinfo = 200
staticgaze_tensorboard_dir = '/home/changfei/GP_static/tensorboard/staticgaze'
staticgaze_weights_dir = '/home/changfei/GP_static/weights/staticgaze'