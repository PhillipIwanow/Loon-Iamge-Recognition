import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=50, embedding_size=128):
        super(SimpleConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 100)
        self.fc_embedding = nn.Linear(100, embedding_size)
        self.fc_softmax = nn.Linear(100, num_classes)

    def forward_sibling(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        embedding = self.fc_embedding(x)
        logits = self.fc_softmax(x)
        return embedding, logits

    def forward(self, input1, input2, input3):
        emb1, logits1 = self.forward_sibling(input1)
        emb2, logits2 = self.forward_sibling(input2)
        emb3, logits3 = self.forward_sibling(input3)
        return emb1, emb2, emb3, torch.cat((logits1, logits2, logits3), 0)

def SimpleConvNetPrototype(num_classes=50, embedding_size=128):
    return SimpleConvNet(num_classes=num_classes, embedding_size=embedding_size)