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
            'mse_reduction': {'value': True},
            'batch_size': {
                # integers between 32 and 256
                # with evenly-distributed logarithms
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 32,
                'max': 256,
            },
            'fe_type': {'value': 'resnet18'},
            'feature_extractor_epochs': {'value': 30},
            'classifier_epochs': {'value': 5},
            'generator_type': {'value': 'gan'},
            'max_generations': {'value': 3},
            'biggan_training': {'value': True},
            'final_task_only': {'value': True},
            'train_aug': {'value': True},
            'skip_normalization': {'value': False},
            'offline_validation': {'value': False},
            'cutmix': {'value': True},
            'log_wandb': {'value': True},
            'gpuid': {'value': 0}
        }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='test_sweep'
)

wandb.agent(sweep_id, function=main, count=2)
