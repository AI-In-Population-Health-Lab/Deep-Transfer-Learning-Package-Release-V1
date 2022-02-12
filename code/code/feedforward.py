from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dalib.modules.classifier import Classifier as ClassifierBase

# used for DANN and MCD
class BackboneNN(nn.Module):
    def __init__(self, input_dim):
        super(BackboneNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out_features = 128
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

# used to give BackboneNN a proper output layer
class BackboneClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BackboneClassifierNN, self).__init__()
        self.bb = BackboneNN(input_dim)
        self.out = nn.Linear(self.bb.out_features, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bb(x))
        return self.out(x)

# used in baseline models
class BackboneClassifierNN_M4(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BackboneClassifierNN_M4, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# used for DANN
class BottleneckNN(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            #nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(BottleneckNN, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)

# used for MCD
class ClassifierHead(nn.Module):
    """
    Classifier head for MCD algorithm.
    parameters:
        in_features (int): input feature dimension
        num_classes (int): number of classes
        bottleneck_dim (int, optional): bottleneck layer dimension; default: 1024
    shapes:
        input: (minibatch, in_features)
        output: (minibatch, num_classes)
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024):
        super(ClassifierHead, self).__init__()
        self.head = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Linear(in_features, bottleneck_dim),
            #nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            #nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)

    def get_parameters(self) -> List[Dict]:
        """
        return:
            params: parameter list which decides optimization hyper-parameters
        """
        params = [{"params": self.head.parameters()},]
        return params
