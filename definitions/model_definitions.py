import gdumb_resnet
from definitions.mlp import MLP, Classifier


def create_feature_extractor(device, latent_size, in_size, args):
    if args.fe_type in ["resnet18", "resnet32"]:
        return gdumb_resnet.ResNet(opt=args).to(device)
        # return torchvision.models.resnet18(pretrained=False, num_classes=latent_size).to(device)
    else:
        return MLP(model_type=args.fe_type,
                   latent_size=latent_size,
                   device=device,
                   in_size=in_size).to(device)


def create_classifier(device, latent_size, n_classes, hidden_size):
    return Classifier(latent_size=latent_size, n_classes=n_classes, hidden_size=hidden_size, device=device)
