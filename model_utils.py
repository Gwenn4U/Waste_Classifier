import torch
import torch.nn as nn
from torchvision import models, transforms


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 64), conv_block(64, 128),
            conv_block(128, 256), conv_block(256, 512),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )

    def forward(self, image):
        return self.classifier(self.features(image))


def build_model(model_name, num_classes, pretrained=False):
    if model_name == 'custom_cnn':
        return CustomCNN(num_classes)

    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
        return model

    if model_name == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.last_channel, num_classes))
        return model

    raise ValueError(f'Unknown model: {model_name}')


def load_model(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(ckpt['model_name'], len(ckpt['classes']), pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt


def get_transform(ckpt):
    size = ckpt['image_size']
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(ckpt['mean'], ckpt['std']),
    ])


@torch.inference_mode()
def predict(model, ckpt, image, device='cpu', top_k=3):
    """image: a PIL.Image. Returns [(class_name, probability), ...]."""
    x = get_transform(ckpt)(image.convert('RGB')).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    top_p, top_i = probs.topk(top_k)
    return [(ckpt['classes'][i], p) for i, p in zip(top_i.tolist(), top_p.tolist())]
