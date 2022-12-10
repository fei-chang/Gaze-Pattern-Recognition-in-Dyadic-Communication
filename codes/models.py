import torch
import torch.nn as nn
from torchvision import models

# load the pre-trained ResNet34 model
resnet34 = models.resnet34(pretrained=True)

# load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# define the basic model structure to pretrain on gazefollow dataset
class GazeFollow_Model(torch.nn.Module):
    """A very simple, naive structure containing only two feature extractors, 
    one ResNet34 for face and the other ResNet50 for the scene with head position stressed.
    The model then concat two features and output a 64x64 heatmap to predict gaze point location.
    """
    def __init__(self):
        super(GazeFollow_Model, self).__init__()
        self.feature_extractor1 = resnet34
        self.feature_extractor2 = resnet50
        self.output = torch.nn.Conv2d(4096, 64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, face_img, scene_img):
        face_feature = self.feature_extractor1(face_img)
        scene_feature = self.feature_extractor2(scene_img)
        x = torch.cat((face_feature, scene_feature), dim=1)
        x = self.output(x)
        return x