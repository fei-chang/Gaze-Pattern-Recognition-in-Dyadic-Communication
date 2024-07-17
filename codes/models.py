import torch
import torch.nn as nn
from torchvision import models
import math

# load the pre-trained ResNet34 model
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

# load the pre-trained ResNet50 model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

class GazeFollow_Model(torch.nn.Module):
    """A very simple, naive structure containing only two feature extractors, 
    one ResNet34 for face and the other ResNet50 for the scene.
    The model then concat two features with the head position embeddings,
    and output a 64x64 heatmap to predict gaze point location.
    """
    def __init__(self):
        super(GazeFollow_Model, self).__init__()
        self.extractor_face = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.extractor_face.fc = nn.Linear(self.extractor_face.fc.in_features, 2048)
        self.extractor_scene = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.extractor_scene.fc = nn.Linear(self.extractor_face.fc.in_features, 2048)
        self.deconv1 = nn.ConvTranspose2d(in_channels=320, out_channels=128, kernel_size=3, stride=5, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)


    def forward(self, face_img, scene_img, head_embeddings):
        face_feature = self.extractor_face(face_img) 
        scene_feature = self.extractor_scene(scene_img)
        x = torch.cat((face_feature, scene_feature, head_embeddings), dim=1)
        x = x.view(x.size(0), 320, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v):
        super(Attention, self).__init__()
        self.norm = 1/math.sqrt(dim_k)
        self.q = nn.Linear(input_dim, dim_k) 
        self.k = nn.Linear(input_dim, dim_k) 
        self.v = nn.Linear(input_dim, dim_v) 
    
    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        att = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)))/self.norm
        out = torch.bmm(att, V)
        
        return out


class GazePattern_Model_V1(torch.nn.Module):
    """The gaze pattern model adopted from GazeFollow Model
    This model takes features from pretrained resnet and with an attention module to fuse the features 
    and generate predictions
    """
    def __init__(self):
        super(GazePattern_Model_V1, self).__init__()
        self.extractor_face = resnet34
        self.extractor_face.fc = nn.Linear(resnet34.fc.in_features, 2048)
        self.extractor_scene = resnet50
        self.extractor_scene.fc = nn.Linear(resnet50.fc.in_features, 2048)
        
        self.attention = Attention(2048, 128)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)



    def forward(self, face_imgs, scene_imgs, head_embeddings):
        face_img1 = face_imgs[:, :int(face_imgs.shape[1]/2), :, :]
        face_img2 = face_imgs[:, int(face_imgs.shape[1]/2):, :, :]
          
        face_feature1 = self.extractor_face(face_img1).unsqueeze(1)  # (batch_size x 1 x 2048)
        face_feature2 = self.extractor_face(face_img2).unsqueeze(1)  # (batch_size x 1 x 2048)
        scene_feature = self.extractor_scene(scene_imgs).unsqueeze(1) # (batch_size x 1 x 2048)
        
        
        combined_features = torch.concat([scene_feature, face_feature1, face_feature2, head_embeddings], dim=1)
        merged_feature = self.attention(combined_features)

        out = self.fc1(merged_feature)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class GazePattern_Model_V2(torch.nn.Module):
    """The gaze pattern model adopted from GazeFollow Model
    This model takes features from pretrained resnet and with attention module to fuse the features 
    and generate predictions
    """
    def __init__(self):
        super(GazePattern_Model_V2, self).__init__()
        self.extractor_face = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.extractor_face.fc = nn.Linear(self.extractor_face.fc.in_features, 2048)
        self.extractor_scene = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.extractor_scene.fc = nn.Linear(self.extractor_scene.fc.in_features, 2048)
        
        self.attention = Attention(2048, 128, 256)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, face_imgs, scene_imgs, head_embeddings):
        face_img1 = face_imgs[:, :int(face_imgs.shape[1]/2), :, :]
        face_img2 = face_imgs[:, int(face_imgs.shape[1]/2):, :, :]
          
        face_feature1 = self.extractor_face(face_img1).unsqueeze(1)  # (batch_size x 1 x 2048)
        face_feature2 = self.extractor_face(face_img2).unsqueeze(1)  # (batch_size x 1 x 2048)
        scene_feature = self.extractor_scene(scene_imgs).unsqueeze(1) # (batch_size x 1 x 2048)
        
        
        combined_features = torch.concat([scene_feature, face_feature1, face_feature2, head_embeddings], dim=1)
        merged_feature = self.attention(combined_features)
        merged_feature = merged_feature.flatten(1)
        out = self.fc1(merged_feature)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class GazePattern_Model_Full(torch.nn.Module):
    """The gaze pattern model
    This model takes features from pretrained resnet and with attention module to fuse the features 
    and generate predictions
    """
    def __init__(self):
        super(GazePattern_Model_Full, self).__init__()
        self.extractor_face = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.extractor_face.fc = nn.Linear(self.extractor_face.fc.in_features, 2048)
        self.extractor_scene = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.extractor_scene.fc = nn.Linear(self.extractor_scene.fc.in_features, 2048)
        
        self.attention = Attention(2048, 128, 256)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, face_imgs, scene_imgs, head_embeddings):
        face_img1 = face_imgs[:, :int(face_imgs.shape[1]/2), :, :]
        face_img2 = face_imgs[:, int(face_imgs.shape[1]/2):, :, :]
          
        face_feature1 = self.extractor_face(face_img1).unsqueeze(1)  # (batch_size x 1 x 2048)
        face_feature2 = self.extractor_face(face_img2).unsqueeze(1)  # (batch_size x 1 x 2048)
        scene_feature = self.extractor_scene(scene_imgs).unsqueeze(1) # (batch_size x 1 x 2048)
        
        
        combined_features = torch.concat([scene_feature, face_feature1, face_feature2, head_embeddings], dim=1)
        merged_feature = self.attention(combined_features)
        merged_feature = merged_feature.flatten(1)
        out = self.fc1(merged_feature)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


