import torch.nn as nn
import torchvision.models as models

class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)

        # Optional: Freeze all except last layer (for small datasets)
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)

    def forward(self, x):
        return self.base_model(x)


