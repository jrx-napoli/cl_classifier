import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset_gen
import training_boot
import utils
import validation
import wandb
from definitions import model_definitions
from options import get_args


def run(args):

    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")

    if args.seed:
        print("Using manual seed = {}".format(args.seed))

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    # if args.log_wandb:
    #     wandb.init(project=f"cl_classifier_{args.experiment_name}")

    # Get transformed data
    train_dataset, val_dataset = dataset_gen.__dict__[args.dataset](args.dataroot,
                                                                    args.skip_normalization,
                                                                    args.train_aug)

    # Prepare dataloaders
    # TODO add 1 class only eval datasets
    train_loaders, _ = dataset_gen.get_CI_datasplit(dataset=train_dataset,
                                                    n_tasks=args.n_tasks,
                                                    n_classes_per_task=args.n_classes_per_task,
                                                    batch_size=args.batch_size,
                                                    drop_last=True)

    val_loaders, val_datasets = dataset_gen.get_CI_datasplit(dataset=val_dataset,
                                                             n_tasks=args.n_tasks,
                                                             n_classes_per_task=args.n_classes_per_task,
                                                             batch_size=args.batch_size,
                                                             drop_last=False)

    ci_eval_dataloaders = dataset_gen.get_CI_eval_dataloaders(val_dataset_splits=val_datasets,
                                                              n_tasks=args.n_tasks,
                                                              batch_size=args.batch_size)

    # Calculate constants
    task_names = [i for i in range(args.n_tasks)]
    print(f'\nTask order: {task_names}')

    # Accuracy tracking
    global_accuracies = []
    x, accuracy = utils.prepare_accuracy_data(n_tasks=args.n_tasks)

    # Prepare models
    input_size = train_dataset[0][0].size()[1]
    translated_latent_size = utils.calculate_translated_latent_size(args=args)
    feature_extractor = model_definitions.create_feature_extractor(device=device,
                                                                   latent_size=translated_latent_size,
                                                                   in_size=input_size,
                                                                   fe_type=args.fe_type)
    classifier = model_definitions.create_classifier(device=device,
                                                     latent_size=translated_latent_size,
                                                     n_classes=args.num_classes,
                                                     hidden_size=translated_latent_size)
    if args.offline_validation:
        validator = validation.OfflineArchitectureValidator()
        validator.test_architecture(args, feature_extractor, classifier, device)
        return

    print(f'\nPrepared models:')
    print(feature_extractor)
    print(classifier)

    # Experiment loop
    for task_id in range(args.n_tasks):

        # skip all non final tasks
        if args.final_task_only and task_id != (args.n_tasks - 1):
            continue

        train_loader = train_loaders[task_id]

        print(f'\n######### Task number {task_id} #########')
        feature_extractor, classifier = training_boot.train_classifier(args=args,
                                                                       feature_extractor=feature_extractor,
                                                                       classifier=classifier,
                                                                       train_loader=train_loader,
                                                                       task_id=task_id,
                                                                       device=device)

        # Calculate current total accuracy
        total_acc = validation.validate_classifier(feature_extractor=feature_extractor,
                                                   classifier=classifier,
                                                   data_loader=ci_eval_dataloaders[task_id])
        print(f'Total accuracy: {total_acc} %')
        global_accuracies.append(total_acc)
        if args.log_wandb:
            wandb.log({"Total accuracy": total_acc})

        # Validate Classifier on all tasks
        for i in range(task_id + 1):
            val_dataset_loader = val_loaders[i]
            acc = validation.validate_classifier(feature_extractor=feature_extractor,
                                                 classifier=classifier,
                                                 data_loader=val_dataset_loader)
            print(f'Task {i}: {acc} %')
            accuracy[i].append(acc)
            if args.log_wandb and not args.final_task_only:
                wandb.log({f"Accuracy on task {i}": acc})

        # Global accuracy graph
        # if task_id == task_names[-1] and not args.final_task_only:
        #     plt.plot(task_names, global_accuracies)
        #     plt.title("Global accuracy")
        #     plt.xlabel("Task id")
        #     plt.ylabel("Accuracy")
        #     ax = plt.gca()
        #     ax.set_ylim([0, 100])
        #     plt.show()
        #
        # # Per-task accuracy graph
        # if task_id == task_names[-1] and not args.final_task_only:
        #     for j in range(len(task_names)):
        #         plt.plot(x[j], accuracy[j])
        #     ax = plt.gca()
        #     ax.set_ylim([0, 100])
        #     plt.title("Accuracy per task")
        #     plt.xlabel("Task id")
        #     plt.ylabel("Accuracy")
        #     plt.show()

    return total_acc


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")

    if args.seed:
        print("Using manual seed = {}".format(args.seed))

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    run(args)
