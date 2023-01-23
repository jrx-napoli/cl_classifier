import sys
import argparse
import copy
import random
import torch, torchvision
from torchvision import transforms
import torch.utils.data as data
from random import shuffle
from collections import OrderedDict
import matplotlib.pyplot as plt
import continual_benchmark.dataloaders.base
import continual_benchmark.dataloaders as dataloaders
from continual_benchmark.dataloaders.datasetGen import data_split
from vae_experiments import classifier_utils
from vae_experiments import multiband_training, classifier_training, replay_training, training_functions
from vae_experiments import vae_utils
from vae_experiments.validation import Validator, CERN_Validator
from vae_experiments import models_definition, classifier_models
from visualise import *
import wandb


def run(args):

    wandb.init(project=f"cl_classifier_{args.experiment_name}")

    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.skip_normalization,
                                                                         args.train_aug)

    if args.dataset.lower() == "celeba":
        n_classes = 10
    else:
        n_classes = train_dataset.number_classes

    n_batches = args.num_batches
    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=n_batches,
                                                                             num_classes=n_classes,
                                                                             random_split=args.random_split,
                                                                             random_mini_shuffle=args.random_shuffle,
                                                                             limit_data=args.limit_data,
                                                                             dirichlet_split_alpha=args.dirichlet,
                                                                             reverse=args.reverse,
                                                                             limit_classes=args.limit_classes)

        

    # Calculate constants
    labels_tasks = {}
    for task_name, task in train_dataset_splits.items():
        labels_tasks[int(task_name)] = task.dataset.class_list

    n_tasks = len(labels_tasks)
    print(f'labels_tasks: {labels_tasks}')
    
    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)
    fid_table = OrderedDict()
    precision_table = OrderedDict()
    recall_table = OrderedDict()
    test_fid_table = OrderedDict()
    fid_local_vae = OrderedDict()


    if args.training_procedure == "classifier":
        # Prepare classifier models
        feature_extractor = classifier_models.FeatureExtractor(latent_size=args.gen_latent_size, 
                                                                d=args.gen_d, 
                                                                cond_dim=n_classes, 
                                                                cond_p_coding=args.gen_cond_p_coding, 
                                                                cond_n_dim_coding=args.gen_cond_n_dim_coding, 
                                                                device=device, 
                                                                in_size=train_dataset[0][0].size()[1], 
                                                                fc=args.fc).to(device)    
        print(feature_extractor)

        classifier = classifier_models.Head(latent_size=args.gen_latent_size, 
                                            d=args.gen_d, 
                                            device=device, 
                                            in_size=train_dataset[0][0].size()[1], 
                                            fc=args.fc).to(device)
        print(classifier)

        # for classifiers accuracy validation chart purpose
        global_accuracies = []
        x = []
        accuracy = []
        for task in task_names:
            accuracy.append([])
            x.append([])
            for i, _ in enumerate(x):
                x[i].append(task)
        
        global_eval_dataloaders = classifier_utils.get_global_eval_dataloaders(task_names=task_names, 
                                                                                val_dataset_splits=val_dataset_splits, 
                                                                                args=args)

    elif args.training_procedure == "multiband":
        # Prepare VAE
        local_vae = models_definition.VAE(latent_size=args.gen_latent_size, 
                                        binary_latent_size=args.binary_latent_size,
                                        d=args.gen_d,
                                        p_coding=args.gen_p_coding,
                                        n_dim_coding=args.gen_n_dim_coding, cond_p_coding=args.gen_cond_p_coding,
                                        cond_n_dim_coding=args.gen_cond_n_dim_coding, cond_dim=n_classes,
                                        device=device, standard_embeddings=args.standard_embeddings,
                                        trainable_embeddings=args.trainable_embeddings,
                                        fc=args.fc,
                                        in_size=train_dataset[0][0].size()[1]).to(device)
        print(local_vae)

    translate_noise = True
    class_table = torch.zeros(n_tasks, n_classes, dtype=torch.long)


    train_loaders = []
    train_loaders_big = []
    val_loaders = []

    # Prepare dataloaders
    for task_name in range(n_tasks):
        train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[task_name],
                                               batch_size=args.gen_batch_size, shuffle=True,
                                               drop_last=False)
        train_dataset_loader_big = data.DataLoader(dataset=train_dataset_splits[task_name],
                                                   batch_size=args.generations_for_switch, shuffle=True,
                                                   drop_last=False)
        train_loaders.append(train_dataset_loader)
        train_loaders_big.append(train_dataset_loader_big)
        val_data = val_dataset_splits[task_name] if args.score_on_val else train_dataset_splits[task_name]
        val_loader = data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)
        val_loaders.append(val_loader)

    if args.dirichlet != None:
        labels_tasks_str = "_".join(["_".join(str(label) for label in labels_tasks[task]) for task in labels_tasks])
        labels_tasks_str = labels_tasks_str[:min(20, len(labels_tasks_str))]
    else:
        labels_tasks_str = ""
    if not args.skip_validation:
        stats_file_name = f"seed_{args.seed}_batches_{args.num_batches}_labels_{labels_tasks_str}_val_{args.score_on_val}_random_{args.random_split}_shuffle_{args.random_shuffle}_dirichlet_{args.dirichlet}_limit_{args.limit_data}"
        if args.dataset.lower() != "cern":
            validator = Validator(n_classes=n_classes, device=device, dataset=args.dataset,
                                  stats_file_name=stats_file_name,
                                  score_model_device=device, dataloaders=val_loaders)
        else:
            validator = CERN_Validator(dataloaders=val_loaders, stats_file_name=stats_file_name, device=device)
    curr_global_decoder = None
    

    for task_id in range(len(task_names)):

        print("\n######### Task number {} #########".format(task_id))
        task_name = task_names[task_id]

        train_dataset_loader = train_loaders[task_id]
        train_dataset_loader_big = train_loaders_big[task_id]

        if args.training_procedure == "multiband":
            curr_global_decoder = multiband_training.train_multiband(args=args, models_definition=models_definition,
                                                                    local_vae=local_vae,
                                                                    curr_global_decoder=curr_global_decoder,
                                                                    task_id=task_id,
                                                                    train_dataset_loader=train_dataset_loader,
                                                                    train_dataset_loader_big=train_dataset_loader_big,
                                                                    class_table=class_table, 
                                                                    n_classes=n_classes,
                                                                    device=device)
        elif args.training_procedure == "replay":
            curr_global_decoder, tmp_table = replay_training.train_with_replay(args=args, 
                                                                                local_vae=local_vae,
                                                                                task_loader=train_dataset_loader,
                                                                                train_dataset_loader_big=train_dataset_loader_big,
                                                                                task_id=task_id, 
                                                                                class_table=class_table)
            class_table[task_id] = tmp_table
        elif args.training_procedure == "classifier":
            feature_extractor, classifier = classifier_training.train_classifier(args=args, 
                                                                                feature_extractor=feature_extractor,
                                                                                classifier=classifier,
                                                                                task_id=task_id,
                                                                                device=device)
        else:
            print("Wrong training procedure")
            return None
        
        # if not using pretrained models, save multiband 
        if not args.gen_load_pretrained_models and args.training_procedure == "multiband":
            torch.save(local_vae, f"results/{args.generator_type}/{args.experiment_name}/model{task_id}_local_vae")
            torch.save(curr_global_decoder, f"results/{args.generator_type}/{args.experiment_name}/model{task_id}_curr_decoder")


        # save feature extractor and classifier
        if args.training_procedure == "classifier":
            torch.save(feature_extractor, f"results/{args.generator_type}/{args.experiment_name}/model{task_id}_feature_extractor")
            torch.save(classifier, f"results/{args.generator_type}/{args.experiment_name}/model{task_id}_classifier")

            cv = classifier_utils.ClassifierValidator()

            # Calculate global accuracy for current task
            correct, total = cv.validate_classifier(fe=feature_extractor, 
                                                    classifier=classifier, 
                                                    data_loader=global_eval_dataloaders[task_id])             
            acc = np.round(100 * correct/total, 3)
            print(f'Global accuracy: {acc} %')
            wandb.log({"Global accuracy": (acc)})
            global_accuracies.append(acc)

            # At the end of the training display global accuracy graph 
            if task_id == task_names[-1]:
                plt.plot(task_names, global_accuracies)
                plt.title("Global classifier accuracy")
                plt.xlabel("Task id")
                plt.ylabel("Accuracy")
                ax = plt.gca()
                ax.set_ylim([0, 100])
                plt.show()

            # Validate Feature Extractor on all tasks
            if args.calc_cosine_similarity == True:
                print(f'\nFeature extractor\'s mean cosine similarity:') 
                for i in range(task_id + 1):
                    val_dataset_loader = val_loaders[i]
                    result = cv.validate_feature_extractor(encoder=local_vae.encoder, 
                                                            translator=curr_global_decoder.translator, 
                                                            feature_extractor=feature_extractor, 
                                                            data_loader=val_dataset_loader)
                    print(f'Task {i}: {result}')
        
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
            if task_id == task_names[-1]:

                for j in range(len(task_names)):
                    plt.plot(x[j], accuracy[j])

                ax = plt.gca()
                ax.set_ylim([0, 100])
                plt.title("Accuracy per task")
                plt.xlabel("Task id")
                plt.ylabel("Accuracy")
                plt.show()


        fid_table[task_name] = OrderedDict()
        precision_table[task_name] = OrderedDict()
        recall_table[task_name] = OrderedDict()
        if args.skip_validation:
            for j in range(task_id + 1):
                fid_table[j][task_name] = -1
        else:
            if (args.training_procedure == "multiband") and (not args.gen_load_pretrained_models):
                fid_result, precision, recall = validator.calculate_results(curr_global_decoder=local_vae.decoder,
                                                                            class_table=class_table,
                                                                            task_id=task_id, translate_noise=translate_noise,
                                                                            starting_point=local_vae.starting_point,
                                                                            dataset=args.dataset)
                fid_local_vae[task_id] = fid_result
                print(f"FID local VAE: {fid_result}")
            for j in range(task_id + 1):
                val_name = task_names[j]
                print('validation split name:', val_name)
                fid_result, precision, recall = validator.calculate_results(curr_global_decoder=curr_global_decoder,
                                                                            class_table=curr_global_decoder.class_table,
                                                                            task_id=j,
                                                                            translate_noise=translate_noise,
                                                                            dataset=args.dataset)  # task_id != 0)
                fid_table[j][task_name] = fid_result
                precision_table[j][task_name] = precision
                recall_table[j][task_name] = recall
                print(f"FID task {j}: {fid_result}")
        local_vae.decoder = copy.deepcopy(curr_global_decoder)
    return fid_table, task_names, test_fid_table, precision_table, recall_table, fid_local_vae


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--rpath', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--score_on_val', action='store_true', required=False, default=False,
                        help="Compute FID on validation dataset instead of validation dataset")
    parser.add_argument('--val_batch_size', type=int, default=250)
    parser.add_argument('--skip_validation', default=False, action='store_true') # altered
    parser.add_argument('--training_procedure', type=str, default="multiband",
                        help='Training procedure multiband|replay') # altered

    # Data
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CelebA")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    # parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--random_split', dest='random_split', default=False, action='store_true',
                        help="Randomize data in splits")
    parser.add_argument('--limit_data', type=float, default=None,
                        help="limit_data to given %")
    parser.add_argument('--random_shuffle', dest='random_shuffle', default=False, action='store_true',
                        help="Move part of data to next batch")
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--skip_normalization', action='store_true', help='Loads dataset without normalization')
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")
    parser.add_argument('--dirichlet', default=None, type=float,
                        help="Alpha parameter for dirichlet data split")
    parser.add_argument('--reverse', dest='reverse', default=False, action='store_true',
                        help="Reverse the ordering of batches")
    parser.add_argument('--limit_classes', type=int, default=-1)

    # Generative network - multiband vae
    parser.add_argument('--gen_batch_size', type=int, default=64)
    parser.add_argument('--local_lr', type=float, default=0.001)
    parser.add_argument('--local_scheduler_rate', type=float, default=0.99)
    parser.add_argument('--scale_local_lr', default=False, action='store_true',
                        help="Scale lr of local model based on the reconstruction error")
    parser.add_argument('--lap_loss', default=False, action='store_true')
    parser.add_argument('--scale_reconstruction_loss', type=float, default=1)
    parser.add_argument('--global_lr', type=float, default=0.0001)
    parser.add_argument('--global_scheduler_rate', type=float, default=0.99)
    parser.add_argument('--gen_n_dim_coding', type=int, default=4,
                        help="Number of bits used to code task id in binary autoencoder")
    parser.add_argument('--gen_p_coding', type=int, default=9,
                        help="Prime number used to calculated codes in binary autoencoder")
    parser.add_argument('--gen_cond_n_dim_coding', type=int, default=0,
                        help="Number of bits used to code task id in binary autoencoder")
    parser.add_argument('--gen_cond_p_coding', type=int, default=9,
                        help="Prime number used to calculated codes in binary autoencoder")
    parser.add_argument('--gen_latent_size', type=int, default=10, help="Latent size in VAE")
    parser.add_argument('--binary_latent_size', type=int, default=4, help="Binary latent size in VAE")
    parser.add_argument('--gen_d', type=int, default=8, help="Size of binary autoencoder")
    parser.add_argument('--gen_ae_epochs', type=int, default=70,
                        help="Number of epochs to train local variational autoencoder")
    parser.add_argument('--global_dec_epochs', type=int, default=140, help="Number of epochs to train global decoder")
    parser.add_argument('--gen_load_pretrained_models', default=False, action='store_true', 
                        help="Load pretrained generative models")
    parser.add_argument('--gen_pretrained_models_dir', type=str, default="results/MNIST_example/",
                        help="Directory of pretrained generative models")
    parser.add_argument('--standard_embeddings', dest='standard_embeddings', default=False, action='store_true',
                        help="Train multiband with standard embeddings instead of matrix")
    parser.add_argument('--trainable_embeddings', dest='trainable_embeddings', default=False, action='store_true',
                        help="Train multiband with trainable embeddings instead of matrix")
    parser.add_argument('--fc', default=False, action='store_true',
                        help="Use only dense layers in VAE model")
    parser.add_argument('--cosine_sim', default=1.0, type=float,
                        help="Cosine similarity between examples to merge")
    parser.add_argument('--limit_previous', default=0.5, type=float,
                        help="How much of previous data we want to generate each epoch")
    parser.add_argument('--global_warmup', default=5, type=int,
                        help="Number of epochs for global warmup - only translator training")
    parser.add_argument('--generations_for_switch', default=1000, type=int,
                        help="Number of noise instances we want to create in order to select instances pos")
    parser.add_argument('--visualise_latent', default=False, action='store_true',
                        help="Whether to visualise latent space")

    # classifier
    parser.add_argument('--generator_type', type=str, default="vae",
                        help='vae|gan')
    parser.add_argument('--gen_load_feature_extractor', default=False, action='store_true',
                        help="Load Feature Extractor")
    parser.add_argument('--gen_load_classifier', default=False, action='store_true',
                        help="Load Classifier")
    parser.add_argument('--feature_extractor_epochs', default=45, type=int,
                        help="Feature Extractor training epochs")
    parser.add_argument('--classifier_epochs', default=4, type=int,
                        help="Classifier training epochs")
    parser.add_argument('--global_benchmark', default=False, action='store_true',
                        help="Train a global classifier as a benchmark model")
    parser.add_argument('--calc_cosine_similarity', default=False, action='store_true',
                        help="During validation, calculate feature extractors cosine similarity")
    

    args = parser.parse_args(argv)

    if args.trainable_embeddings:
        args.standard_embeddings = True

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

    acc_val, acc_test, precision_table, recall_table = {}, {}, {}, {}
    os.makedirs(f"{args.rpath}{args.experiment_name}", exist_ok=True)
    with open(f"{args.rpath}{args.experiment_name}/args.txt", "w") as text_file:
        text_file.write(str(args))
    for r in range(args.repeat):
        acc_val[r], _, acc_test[r], precision_table[r], recall_table[r], fid_local_vae = run(args)
    np.save(f"{args.rpath}{args.experiment_name}/fid.npy", acc_val)
    np.save(f"{args.rpath}{args.experiment_name}/precision.npy", precision_table)
    np.save(f"{args.rpath}{args.experiment_name}/recall.npy", recall_table)
    np.save(f"{args.rpath}{args.experiment_name}/fid_local_vae.npy", fid_local_vae)
    plot_final_results([args.experiment_name], type="fid", fid_local_vae=fid_local_vae)
    print(fid_local_vae)
