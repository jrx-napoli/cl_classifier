from definitions.gdumb_resnet import ResNet
from definitions.continual_benchmark_resnet import ResNet18, ResNet34
from definitions.mlp import MLP, Classifier
import torchvision


def create_feature_extractor(device, latent_size, in_size, args):
    if args.fe_type == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=latent_size).to(device)
        # return ResNet(opt=args).to(device)
        # return ResNet18(out_dim=latent_size).to(device)
    elif args.fe_type == "resnet34":
        return ResNet34(out_dim=latent_size).to(device)
    else:
        return MLP(model_type=args.fe_type,
                   latent_size=latent_size,
                   device=device,
                   in_size=in_size).to(device)


def create_classifier(device, latent_size, n_classes, hidden_size):
    return Classifier(latent_size=latent_size, n_classes=n_classes, hidden_size=hidden_size, device=device)
