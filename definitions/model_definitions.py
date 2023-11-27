from definitions.mlp import MLP
from definitions.conv import Conv
from definitions.classifier import Classifier
import torchvision


def create_feature_extractor(device, latent_size, in_size, fe_type):
    if fe_type == "mlp400":
        return MLP(latent_size=latent_size,
                   in_size=in_size).to(device)
    elif fe_type == "conv":
        return Conv(latent_size=latent_size).to(device)
    elif fe_type == "resnet18":
        return torchvision.models.resnet18(num_classes=latent_size).to(device)
    elif fe_type == "resnet34":
        return torchvision.models.resnet34(num_classes=latent_size).to(device)


def create_classifier(latent_size, n_classes, hidden_size, device):
    return Classifier(latent_size=latent_size, n_classes=n_classes, hidden_size=hidden_size, device=device).to(device)
