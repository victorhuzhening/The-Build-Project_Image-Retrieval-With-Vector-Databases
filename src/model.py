import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGFineTuned(nn.Module):
    """
    VGG-16 backbone → (features + avgpool) → flatten (25088 dims) → 128-D embedding → classifier.
    Early conv layers are mostly frozen; last few are trainable.
    """
    def __init__(self, num_classes=101, embedding_size=128, pretrained=True, unfreeze_last_n_params=20):
        super().__init__()

        # Load VGG-16
        self.vgg = models.vgg11(pretrained=pretrained)

        # Freeze most of the convolutional backbone; leave the last N params trainable
        for p in self.vgg.features.parameters():
            p.requires_grad = False
        # Unfreeze the last N conv params (fine-tuning tail)
        trainable = list(self.vgg.features.parameters())[-unfreeze_last_n_params:]
        for p in trainable:
            p.requires_grad = True

        self.vgg.classifier = nn.Identity() # not original classifier head because it's too expensive

        # VGG flow is: features -> avgpool -> flatten -> classifier
        # After avgpool + flatten, VGG-16 produces 25088 dims for 224x224 inputs.
        feature_dim = 25088

        # Our embedding head
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )

        # Final classifier
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # VGG forward up to features
        x = self.vgg.features(x)        # [B, 512, 7, 7] for 224x224
        x = self.vgg.avgpool(x)         # adaptive avgpool -> [B, 512, 7, 7]
        x = torch.flatten(x, 1)         # [B, 25088]
        emb = self.embedding(x)         # [B, embedding_size]
        logits = self.classifier(emb)   # [B, num_classes]
        return logits

    @torch.no_grad()
    def extract_features(self, x):
        self.eval()
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        return F.normalize(emb, p=2, dim=1)
