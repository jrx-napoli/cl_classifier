import math
import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim

from multiband_vae.vae_experiments import vae_utils
from multiband_vae.gan_experiments import gan_utils
import wandb

torch.autograd.set_detect_anomaly(True)


def train_feature_extractor(args, feature_extractor, decoder, task_id, device,
                            local_translator_emb_cache=None, train_loader=None, local_start_lr=0.001, scheduler_rate=0.99):
    wandb.watch(feature_extractor)
    feature_extractor.train()
    decoder.translator.eval()
    decoder.eval()

    n_epochs = args.feature_extractor_epochs
    batch_size = args.gen_batch_size

    if train_loader:
        n_iterations = len(train_loader)
        n_prev_examples = int(batch_size * min(task_id, 7))
        n_tasks = task_id
    else:
        n_iterations = 100 # TODO parametrise this
        n_prev_examples = int(batch_size * min(task_id + 1, 7))
        n_tasks = task_id + 1
    lr = local_start_lr

    print(f'Iterations: {n_iterations}')
    print(f'Generations per iteration: {n_prev_examples}')
    print(f"Feature extractor's lr set to: {lr}")

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    
    for epoch in range(n_epochs):
        losses = []
        cosine_distances = []
        start = time.time()

        # Optimise noise and store it for entire training process
        if epoch == 0 and args.generator_type == "gan" and local_translator_emb_cache == None:
            print("Noise optimisation for GAN embeddings")
            for iteration, batch in enumerate(train_loader):
                print(f"Batch {iteration}/{n_iterations}")

                local_imgs, local_classes = batch
                local_imgs = local_imgs.to(device)
                local_classes = local_classes.to(device)
                _, local_translator_emb = gan_utils.optimize_noise(images=local_imgs, 
                                                         generator=decoder, 
                                                         n_iterations=500,
                                                         task_id=task_id, # NOTE task_id does not matter in this implementation
                                                         lr=0.01,
                                                         labels=local_classes)
                local_translator_emb = local_translator_emb.detach()

                if local_translator_emb_cache == None:
                    local_translator_emb_cache = local_translator_emb
                else:
                    local_translator_emb_cache = torch.cat((local_translator_emb_cache, local_translator_emb), 0)
                
            torch.save(local_translator_emb_cache, f"results/{args.generator_type}/{args.experiment_name}/local_translator_emb_cache_BETA")
                

        for iteration, batch in enumerate(train_loader):

            # local data
            local_imgs, local_classes = batch
            local_imgs = local_imgs.to(device)
            # local_classes = local_classes.to(device)

            emb_start_point = iteration * batch_size
            emb_end_point = min(len(train_loader.dataset), (iteration + 1) * batch_size)
            local_translator_emb = local_translator_emb_cache[emb_start_point:emb_end_point]

            # rehearsal data
            with torch.no_grad():

                if args.generator_type == "vae":
                    generations, _, _, translator_emb = vae_utils.generate_previous_data(
                        decoder,
                        n_tasks=n_tasks,
                        n_img=n_prev_examples,
                        num_local=batch_size,
                        return_z=True,
                        translate_noise=True,
                        same_z=False)

                elif args.generator_type == "gan":
                    # TODO bugfix: number of generations is rounded down
                    generations, _, classes, translator_emb = gan_utils.generate_previous_data(
                        n_prev_tasks=(5*n_tasks), # TODO adjust for cifar100 example - x5 for 20 tasks?
                        n_prev_examples=n_prev_examples,
                        curr_global_generator=decoder)
                
                generations = generations.to(device)


            # concat local and generated data
            generations = torch.cat([generations, local_imgs])
            translator_emb = torch.cat([translator_emb, local_translator_emb])
            # classes = torch.cat([classes, local_classes])
            # print(torch.unique(classes, return_counts=True))

            # shuffle
            n_mini_batches = math.ceil(len(generations) / batch_size)
            shuffle = torch.randperm(len(generations))
            generations = generations[shuffle]
            translator_emb = translator_emb[shuffle]


            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(generations), (batch_id + 1) * batch_size)

                optimizer.zero_grad()

                out = feature_extractor(generations[start_point:end_point])
                loss = criterion(out, translator_emb[start_point:end_point])

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    losses.append(loss.item())
                    for i, output in enumerate(out):
                        cosine_distances.append((torch.cosine_similarity(output, translator_emb[start_point:end_point][i], dim=0)).item())

        scheduler.step()

        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, cosine similarity: {}, took: {} s".format(epoch, 
                                                                                     n_epochs,
                                                                                     np.round(np.mean(losses), 3),
                                                                                     np.round(np.mean(cosine_distances), 3),
                                                                                     np.round(time.time() - start), 3))

    return feature_extractor, local_translator_emb_cache


def train_head(args, classifier, decoder, task_id, device, train_loader=None, local_translator_emb_cache=None,
               train_same_z=False, local_start_lr=0.001, scheduler_rate=0.99):
    wandb.watch(classifier)
    decoder.translator.eval()
    decoder.eval()
    classifier.train()

    n_epochs = args.classifier_epochs
    batch_size = args.gen_batch_size

    if train_loader:
        n_iterations = len(train_loader)
        n_prev_examples = int(batch_size * min(task_id, 2))
        n_tasks = task_id
    else:
        n_iterations = 100 # TODO parametrise this
        n_prev_examples = int(batch_size * min(task_id + 1, 2))
        n_tasks = task_id + 1
    lr = local_start_lr

    print(f'Iterations: {n_iterations}')
    print(f'Generations per iteration: {n_prev_examples}')
    print(f"Classifier's lr set to: {lr}")

    optimizer = torch.optim.Adam(list(classifier.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        losses = []
        accuracy = 0
        total = 0
        start = time.time()
        
        for iteration, batch in enumerate(train_loader):

            # local data
            _, local_classes = batch
            local_classes = local_classes.to(device)

            emb_start_point = iteration * batch_size
            emb_end_point = min(len(train_loader.dataset), (iteration + 1) * batch_size)
            local_translator_emb = local_translator_emb_cache[emb_start_point:emb_end_point]


            if args.generator_type == "vae":
                _, classes, _, translator_emb = vae_utils.generate_previous_data(
                    decoder,
                    n_tasks=n_tasks,
                    n_img=n_prev_examples,
                    num_local=batch_size,
                    return_z=True,
                    translate_noise=True,
                    same_z=False)

            elif args.generator_type == "gan":
                _, _, classes, translator_emb = gan_utils.generate_previous_data(
                    n_prev_tasks=(5*n_tasks),
                    n_prev_examples=n_prev_examples,
                    curr_global_generator=decoder)

            classes = classes.long()


            # concat local and generated data
            translator_emb = torch.cat([translator_emb, local_translator_emb])
            classes = torch.cat([classes, local_classes])
            # print(torch.unique(classes, return_counts=True))

            # shuffle
            n_mini_batches = math.ceil(len(classes) / batch_size)
            shuffle = torch.randperm(len(classes))
            classes = classes[shuffle]
            translator_emb = translator_emb[shuffle]


            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(classes), (batch_id + 1) * batch_size)

                optimizer.zero_grad()

                out = classifier(translator_emb[start_point:end_point])
                loss = criterion(out, classes[start_point:end_point].to(classifier.device))

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    acc = get_classifier_accuracy(out, classes[start_point:end_point].to(classifier.device))
                    accuracy += acc.item()
                    total += len(classes[start_point:end_point])
                    losses.append(loss.item())

        scheduler.step()

        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, Acc: {} %, took: {} s".format(epoch, n_epochs,
                                                                np.round(np.mean(losses), 3),
                                                                np.round(accuracy * 100 / total, 3),
                                                                np.round(time.time() - start), 3))

    return classifier


def get_classifier_accuracy(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum
