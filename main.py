import sys
import argparse
import copy
import random
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

import dataset_gen
import model_definitions
import training_boot
import validation
import utils

import multiband_vae.continual_benchmark.dataloaders.base
import multiband_vae.continual_benchmark.dataloaders as dataloaders


def run(args):

    if args.log_wandb:
        wandb.init(project=f"cl_classifier_{args.experiment_name}")


    # Get transformed data
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.skip_normalization, args.train_aug)


    # Prepare dataloaders
    train_loaders, train_datasets, n_tasks = dataset_gen.split_data(args=args, dataset=train_dataset, drop_last=True)
    val_loaders, val_datasets, _ = dataset_gen.split_data(args=args, dataset=val_dataset, drop_last=False)
    global_eval_dataloaders = dataset_gen.create_CI_eval_dataloaders(task_names=n_tasks, val_dataset_splits=val_datasets, args=args)
    # TODO add 1 class only eval datasets


    # Calculate constants
    task_names = [i for i in range(n_tasks)]
    print(f'\nTask order: {task_names}')


    # Accuracy tracking
    global_accuracies = []
    x, accuracy = utils.prepare_accuracy_data(n_tasks=n_tasks)


    # Prepare models
    input_size = train_dataset[0][0].size()[1]    
    translated_latent_size = utils.calculate_translated_latent_size(args=args)

    feature_extractor = model_definitions.create_feature_extractor(model_type=args.fe_type, 
                                                                   device=device, 
                                                                   latent_size=translated_latent_size, 
                                                                   in_size=input_size).to(device)
    classifier = model_definitions.create_classifier(device=device, latent_size=translated_latent_size).to(device)
    print(f'\nPrepared models:')
    print(feature_extractor)
    print(classifier)


    # Test calssifier's architecture
    if args.global_benchmark:
        print(f'\nRunning offline benchamrk:')
        feature_extractor_copy = copy.deepcopy(feature_extractor)
        classifier_copy = copy.deepcopy(classifier)

        architecture_validator = validation.OfflineArchitectureValidator()
        architecture_validator.test_architecture(args=args,
                                                 feature_extractor=feature_extractor_copy,
                                                 classifier=classifier_copy,
                                                 device=device)
        return


    for task_id in range(n_tasks):
        if args.final_task_only and task_id != (n_tasks-1):
            # skip all non final tasks
            continue

        if args.train_on_available_data:
            train_loader = train_loaders[task_id]
        else:
            train_loader = None

        print("\n######### Task number {} #########".format(task_id))
        feature_extractor, classifier = training_boot.train_classifier(args=args, 
                                                                       feature_extractor=feature_extractor,
                                                                       classifier=classifier,
                                                                       train_loader=train_loader,
                                                                       task_id=task_id,
                                                                       device=device)
    
        # Save feature extractor and classifier
        if not args.load_feature_extractor:
            torch.save(feature_extractor, f"models/{args.generator_type}/{args.experiment_name}/model{task_id}_feature_extractor")
        
        if not args.load_classifier:
            torch.save(classifier, f"models/{args.generator_type}/{args.experiment_name}/model{task_id}_classifier")


        # Calculate current accuracy
        cv = validation.ClassifierValidator()
        correct, total = cv.validate_classifier(feature_extractor=feature_extractor, 
                                                classifier=classifier, 
                                                data_loader=global_eval_dataloaders[task_id])
        acc = np.round(100 * correct/total, 3)
        print(f'Global accuracy: {acc} %')
        if args.log_wandb:
            wandb.log({"Global accuracy": (acc)})
        global_accuracies.append(acc)


        # At the end of the training display global accuracy graph 
        if task_id == task_names[-1] and not args.final_task_only:
            plt.plot(task_names, global_accuracies)
            plt.title("Global classifier accuracy")
            plt.xlabel("Task id")
            plt.ylabel("Accuracy")
            ax = plt.gca()
            ax.set_ylim([0, 100])
            plt.show()


        # Validate Feature Extractor on all tasks
        if args.calc_cosine_similarity == True:
            # TODO -> implement for both GAN and VAE
            # print(f'\nFeature extractor\'s mean cosine similarity:') 
            # for i in range(task_id + 1):
            #     val_dataset_loader = val_loaders[i]
            #     result = cv.validate_feature_extractor(encoder=local_vae.encoder, 
            #                                             translator=curr_global_decoder.translator, 
            #                                             feature_extractor=feature_extractor, 
            #                                             data_loader=val_dataset_loader)
            #     print(f'Task {i}: {result}')
            pass
    
    
        # Validate Classifer on all tasks
        for i in range(task_id + 1):
            val_dataset_loader = val_loaders[i]
            correct, total = cv.validate_classifier(feature_extractor=feature_extractor, 
                                                    classifier=classifier, 
                                                    data_loader=val_dataset_loader)
            acc = np.round(100 * correct/total, 3)
            print(f'Task {i}: {correct}/{total} ({acc} %)')
            accuracy[i].append(acc)


        # At the end of the training display per-task accuracy graph
        if task_id == task_names[-1] and not args.final_task_only:

            for j in range(len(task_names)):
                plt.plot(x[j], accuracy[j])

            ax = plt.gca()
            ax.set_ylim([0, 100])
            plt.title("Accuracy per task")
            plt.xlabel("Task id")
            plt.ylabel("Accuracy")
            plt.show()
    
    return


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--dataset', type=str, default='MNIST', help="Dataset to be used in training procedure")
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--gen_latent_size', type=int, default=10, help="Latent size in VAE")
    parser.add_argument('--gen_d', type=int, default=8, help="Size of binary autoencoder")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--skip_normalization', action='store_true', help='Loads dataset without normalization')
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--generator_type', type=str, default="vae", help='vae|gan')
    parser.add_argument('--fe_type', type=str, default="mlp400", help='mlp400|conv|resnet18')
    parser.add_argument('--load_feature_extractor', default=False, action='store_true', help="Load Feature Extractor")
    parser.add_argument('--load_classifier', default=False, action='store_true', help="Load Classifier")
    parser.add_argument('--feature_extractor_epochs', default=30, type=int, help="Feature Extractor training epochs")
    parser.add_argument('--classifier_epochs', default=5, type=int, help="Classifier training epochs")
    parser.add_argument('--global_benchmark', default=False, action='store_true', help="Train a global classifier as a benchmark model")
    parser.add_argument('--calc_cosine_similarity', default=False, action='store_true', help="Validate feature extractors cosine similarity")
    parser.add_argument('--reset_model', default=False, action='store_true', help="Reset model before every task")
    parser.add_argument('--final_task_only', default=False, action='store_true', help="Reset model before every task")
    parser.add_argument('--train_on_available_data', default=False, action='store_true', help="Train on available real samples")
    parser.add_argument('--log_wandb', default=False, action='store_true', help="Log training process on wandb")
    
    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")

    if args.seed:
        print("Using manual seed = {}".format(args.seed))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    run(args)
