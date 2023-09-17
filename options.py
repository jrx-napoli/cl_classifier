import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--dataset', type=str, default='MNIST', help="Dataset to be used in training procedure")
    parser.add_argument('--n_tasks', default=5, type=int, help="Number of tasks")
    parser.add_argument('--n_classes_per_task', default=2, type=int, help="Number of classes per task")
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--log_wandb', default=False, action='store_true', help="Log training process on wandb")
    parser.add_argument('--offline_validation', default=False, action='store_true',
                        help="Validate the architecture in an offline manner")

    # Data
    parser.add_argument('--skip_normalization', default=False, action='store_true',
                        help='Loads dataset without normalization')
    parser.add_argument('--train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--cutmix', default=False, action='store_true', help='Use cutmix regularization')
    parser.add_argument('--generations_only', default=False, action='store_true',
                        help='Train only using generated samples')
    parser.add_argument('--max_generations', type=int, default=3,
                        help='Maximum number of generated rehearsal sample batches')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Cutmix alpha parameter')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Cutmix probability')
    parser.add_argument('--global_benchmark', default=False, action='store_true',
                        help="Train a global classifier as a benchmark model")

    # Model
    parser.add_argument('--fe_type', type=str, default="mlp400", help='mlp400|conv|resnet18|resnet32')
    parser.add_argument('--depth', type=int, default=32, help='Depth of a PreAct-Resnet model')
    parser.add_argument('--gen_latent_size', type=int, default=10, help="Latent size in VAE")
    parser.add_argument('--gen_d', type=int, default=8, help="Size of binary autoencoder")
    parser.add_argument('--activetype', default='ReLU',
                        choices=['ReLU6', 'LeakyReLU', 'PReLU', 'ReLU', 'ELU', 'Softplus', 'SELU', 'None'],
                        help='Activation types')
    parser.add_argument('--pooltype', type=str, default='MaxPool2d',
                        choices=['MaxPool2d', 'AvgPool2d', 'adaptive_max_pool2d', 'adaptive_avg_pool2d'],
                        help='Pooling types')
    parser.add_argument('--normtype', type=str, default='BatchNorm', choices=['BatchNorm', 'InstanceNorm'],
                        help='Batch normalization types')
    parser.add_argument('--preact', action="store_true", default=False,
                        help='Places norms and activations before linear/conv layer. Set to False by default')
    parser.add_argument('--bn', action="store_false", default=True, help='Apply Batchnorm. Set to True by default')
    parser.add_argument('--affine_bn', action="store_false", default=True,
                        help='Apply affine transform in BN. Set to True by default')
    parser.add_argument('--bn_eps', type=float, default=1e-6, help='Affine transform for batch norm')
    parser.add_argument('--compression', type=float, default=0.5, help='DenseNet BC hyperparam')
    parser.add_argument('--in_channels', default=3, type=int, help="Number of data channels")
    parser.add_argument('--num_classes', default=0, type=int, help="Number of classes")
    parser.add_argument('--max_lr', type=float, default=0.05, help='Starting Learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0005, help='Ending Learning rate')
    parser.add_argument('--fe_weight_decay', type=float, default=1e-5, help='Feature extractor weight decay')
    parser.add_argument('--cl_weight_decay', type=float, default=1e-5, help='Classifier weight decay')

    # Training
    parser.add_argument('--generator_type', type=str, default="vae", help='vae|gan')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], help='Optimiser types')
    parser.add_argument('--mse_reduction', default=False, action='store_true', help="Use MSE loss reduction type sum")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fe_lr', type=float, default=0.001, help='Feature extractor learning rate')
    parser.add_argument('--cl_lr', type=float, default=0.001, help='Classifier learning rate')
    parser.add_argument('--load_feature_extractor', default=False, action='store_true', help="Load Feature Extractor")
    parser.add_argument('--load_classifier', default=False, action='store_true', help="Load Classifier")
    parser.add_argument('--feature_extractor_epochs', default=30, type=int, help="Feature Extractor training epochs")
    parser.add_argument('--classifier_epochs', default=4, type=int, help="Classifier training epochs")
    parser.add_argument('--calc_cosine_similarity', default=False, action='store_true',
                        help="Validate feature extractors cosine similarity")
    parser.add_argument('--calc_noise', default=False, action='store_true',
                        help="Calculate optimised GAN noise")
    parser.add_argument('--reset_model', default=False, action='store_true', help="Reset model before every task")
    parser.add_argument('--biggan_training', default=False, action='store_true', help="Train on BigGAN generator")
    parser.add_argument('--final_task_only', default=False, action='store_true', help="Reset model before every task")

    return parser.parse_args(argv)
