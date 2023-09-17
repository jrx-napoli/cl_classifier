from main import run

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()


def main():
    wandb.init(project='test_sweep')
    total_accuracy = run(wandb.config)
    wandb.log({'total_accuracy': total_accuracy})


# 2: Define the search space
sweep_configuration = {
    'method': 'random',
    'metric':
        {
            'goal': 'maximize',
            'name': 'total_accuracy'
        },
    'parameters':
        {
            'dataroot': {'value': 'data'},
            'experiment_name': {'value': 'CIFAR10_BIGGAN_example'},
            'dataset': {'value': 'CIFAR10'},
            'num_classes': {'value': 10},
            'n_classes_per_task': {'value': 2},
            'n_tasks': {'value': 5},
            'seed': {'value': 13},
            'optimizer': {'value': 'Adam'},
            'fe_type': {'value': 'resnet18'},
            'generator_type': {'value': 'gan'},
            'mse_reduction': {'value': True},
            'batch_size': {
                # integers between 32 and 256
                # with evenly-distributed logarithms
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 32,
                'max': 256,
            },
            'feature_extractor_epochs': {'values': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]},
            'classifier_epochs': {'values': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]},
            'max_generations': {'values': [1, 2, 3, 4, 5]},
            'load_feature_extractor': {'value': False},
            'load_classifier': {'value': False},
            'generations_only': {'value': False},
            'biggan_training': {'value': True},
            'final_task_only': {'value': True},
            'train_aug': {'value': True},
            'calc_noise': {'value': False},
            'skip_normalization': {'value': False},
            'offline_validation': {'value': False},
            'cutmix': {'values': [True, False]},
            'cutmix_prob': {'values': [0.3, 0.4, 0.5, 0.6]},
            'cutmix_alpha': {'value': 1.0},
            'fe_weight_decay': {'values': [1e-4, 1e-5, 1e-6]},
            'cl_weight_decay': {'values': [1e-4, 1e-5, 1e-6]},
            'fe_lr': {'values': [0.0001, 0.0005, 0.001, 0.005, 0.01]},
            'classifier_lr': {'values': [0.0001, 0.0005, 0.001, 0.005, 0.01]},
            'log_wandb': {'value': True},
            'gpuid': {'value': 0}
        }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='test_sweep'
)

wandb.agent(sweep_id, function=main, count=1)
