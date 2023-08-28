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
            'experiment_name': 'CIFAR10_BIGGAN_example',
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'n_classes_per_task': 2,
            'n_tasks': 5,
            'seed': 13,
            'optimizer': 'Adam',
            'mse_reduction': 'True',
            'batch_size': {
                # integers between 32 and 256
                # with evenly-distributed logarithms
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 32,
                'max': 256,
            },
            'fe_type': 'resnet18',
            'feature_extractor_epochs': 30,
            'classifier_epochs': 5,
            'generator_type': 'gan',
            'max_generations': 3,
            'biggan_training': 'True',
            'final_task_only': 'True',
            'train_aug': 'True',
            'cutmix': 'True',
            'log_wandb': 'True',
            'gpuid': 0
        }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='test_sweep'
)

wandb.agent(sweep_id, function=main, count=10)
