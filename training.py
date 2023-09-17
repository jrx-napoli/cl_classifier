import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import wandb
from gan_experiments import gan_utils
from multiband_vae.vae_experiments import vae_utils
from regularization.cutmix import cutmix_images, cutmix_repr

torch.autograd.set_detect_anomaly(True)


def get_fe_criterion(args):
    if args.mse_reduction:
        return nn.MSELoss(reduction="sum")
    else:
        return nn.MSELoss()


def get_optimiser(args, model):
    if args.optimizer.lower() == "adam":
        return torch.optim.Adam(list(model.parameters()), lr=args.fe_lr, weight_decay=args.fe_weight_decay)
    elif args.optimizer.lower() == "sgd":
        return torch.optim.SGD(list(model.parameters()), lr=args.cl_lr, weight_decay=args.cl_weight_decay)
    else:
        raise NotImplementedError


def get_scheduler(args, optimizer):
    if args.optimizer.lower() == "adam":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.optimizer.lower() == "sgd":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    else:
        raise NotImplementedError


def calculate_gan_noise(args, generator, train_loader, task_id, device):
    print("\nNoise optimisation for GAN embeddings")
    n_iterations = 500
    noise_cache = None

    for iteration, batch in enumerate(train_loader):
        print(f"Batch {iteration}/{len(train_loader)}")

        local_imgs, local_classes = batch
        local_imgs = local_imgs.to(device)
        local_classes = local_classes.to(device)
        local_translator_emb = gan_utils.optimize_noise(images=local_imgs,
                                                        generator=generator,
                                                        n_iterations=n_iterations,
                                                        task_id=task_id,
                                                        lr=0.01,
                                                        labels=local_classes,
                                                        biggan_training=args.biggan_training)
        local_translator_emb = local_translator_emb.detach()

        if noise_cache is None:
            noise_cache = local_translator_emb
        else:
            noise_cache = torch.cat((noise_cache, local_translator_emb), 0)

    torch.save(noise_cache, f"models/{args.generator_type}/{args.experiment_name}/model{task_id}_noise_cache")
    return noise_cache


def generate_images(args, generator, n_prev_examples, task_id):
    if args.generations_only:
        task_id += 1

    if args.generator_type == "vae":
        generations, classes, random_noise, translator_emb = vae_utils.generate_previous_data(
            generator,
            n_tasks=task_id,
            n_img=n_prev_examples,
            num_local=args.batch_size,
            return_z=True,
            translate_noise=True,
            same_z=False)

    elif args.generator_type == "gan":
        # TODO bugfix: number of generations is rounded down
        generations, classes, random_noise, translator_emb = gan_utils.generate_previous_data(
            n_prev_tasks=(2 * task_id),  # TODO adjust for specific dataset - n_classes for each tasks?
            n_prev_examples=n_prev_examples,
            curr_global_generator=generator,
            biggan_training=args.biggan_training)

    return generations, classes, random_noise, translator_emb


def train_feature_extractor(args, feature_extractor, decoder, task_id, device, train_loader,
                            local_vae=None, noise_cache=None):
    if args.log_wandb:
        wandb.watch(feature_extractor)
    feature_extractor.train()
    decoder.translator.eval()
    decoder.eval()

    n_epochs = args.feature_extractor_epochs
    batch_size = args.batch_size
    n_iterations = len(train_loader)

    if args.generations_only:
        n_prev_examples = int(batch_size * args.max_generations)
    else:
        n_prev_examples = int(batch_size * min(task_id + 1, args.max_generations))

    print(f'Iterations /epoch: {n_iterations}')
    print(f'Generations /iteration: {n_prev_examples}')
    if args.log_wandb:
        wandb.run.summary["Iterations per epoch"] = n_iterations
        wandb.run.summary["Generations per iteration"] = n_prev_examples

    criterion = get_fe_criterion(args=args)
    optimizer = get_optimiser(args=args, model=feature_extractor)
    scheduler = get_scheduler(args=args, optimizer=optimizer)

    for epoch in range(n_epochs):
        losses = []
        cosine_similarities = []
        start = time.time()

        for iteration, batch in enumerate(train_loader):

            # if iteration == 5:
            #     break

            # local data
            local_images, local_classes = batch
            local_images = local_images.to(device)

            if args.generator_type == "vae":
                local_translator_emb = local_vae(x=local_images,
                                                 task_id=local_classes,
                                                 conds=None,
                                                 temp=None,
                                                 encode_to_noise=True)
                local_translator_emb = local_translator_emb.detach()
            else:
                emb_start_point = iteration * batch_size
                emb_end_point = min(len(train_loader.dataset), (iteration + 1) * batch_size)
                local_translator_emb = noise_cache[emb_start_point:emb_end_point]

            # rehearsal data
            if args.generations_only or task_id > 0:
                with torch.no_grad():
                    generations, classes, random_noise, translator_emb = generate_images(args=args,
                                                                                         generator=decoder,
                                                                                         n_prev_examples=n_prev_examples,
                                                                                         task_id=task_id)
                    translator_emb = translator_emb.detach()

            # concat local and generated data
            if args.generations_only:
                images_combined = generations
                emb_combined = translator_emb
            elif task_id > 0:
                images_combined = torch.cat([generations, local_images])
                emb_combined = torch.cat([translator_emb, local_translator_emb])
            else:
                images_combined = local_images
                emb_combined = local_translator_emb

            # shuffle
            n_mini_batches = math.ceil(len(images_combined) / batch_size)
            shuffle = torch.randperm(len(images_combined))
            images_combined = images_combined[shuffle]
            emb_combined = emb_combined[shuffle]

            # optimize
            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(images_combined), (batch_id + 1) * batch_size)

                optimizer.zero_grad()

                do_cutmix = args.cutmix and np.random.rand(1) < args.cutmix_prob
                if do_cutmix:
                    inputs, labels_a, labels_b, lam = cutmix_images(x=images_combined[start_point:end_point],
                                                                    y=emb_combined[start_point:end_point],
                                                                    alpha=args.cutmix_alpha)
                    out = feature_extractor(inputs)
                    loss = lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)
                else:
                    out = feature_extractor(images_combined[start_point:end_point])
                    loss = criterion(out, emb_combined[start_point:end_point])

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    losses.append(loss.item())
                    for i, output in enumerate(out):
                        cosine_similarities.append(
                            (torch.cosine_similarity(output, emb_combined[start_point:end_point][i], dim=0)).item())

        scheduler.step()

        if args.log_wandb:
            wandb.log({"Feature-extractor loss": np.round(np.mean(losses), 3)})
            wandb.log({"Mean cosine similarity": np.round(np.mean(cosine_similarities), 3)})
        print("Epoch: {}/{}, loss: {}, cosine similarity: {}, took: {} s".format(epoch,
                                                                                 n_epochs,
                                                                                 np.round(np.mean(losses), 3),
                                                                                 np.round(
                                                                                     np.mean(cosine_similarities), 3),
                                                                                 np.round(time.time() - start), 3))

    return feature_extractor, noise_cache


def train_classifier(args, classifier, decoder, task_id, device, train_loader,
                     local_vae=None, noise_cache=None):
    if args.log_wandb:
        wandb.watch(classifier)
    decoder.translator.eval()
    decoder.eval()
    classifier.train()

    n_epochs = args.classifier_epochs
    batch_size = args.batch_size
    n_iterations = len(train_loader)

    if args.generations_only:
        n_prev_examples = int(batch_size * args.max_generations)
    else:
        n_prev_examples = int(batch_size * min(task_id + 1, args.max_generations))

    print(f'Iterations /epoch: {n_iterations}')
    print(f'Generations /iteration: {n_prev_examples}')

    optimizer = torch.optim.Adam(list(classifier.parameters()), lr=0.001, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        losses = []
        accuracy = 0
        total = 0
        start = time.time()

        for iteration, batch in enumerate(train_loader):

            # if iteration == 5:
            #     break

            # local data
            local_images, local_classes = batch
            local_images = local_images.to(device)

            if args.generator_type == "vae":
                local_translator_emb = local_vae(x=local_images,
                                                 task_id=local_classes,
                                                 conds=None,
                                                 temp=None,
                                                 encode_to_noise=True)
                local_translator_emb = local_translator_emb.detach()
            else:
                local_classes = local_classes.to(device)
                emb_start_point = iteration * batch_size
                emb_end_point = min(len(train_loader.dataset), (iteration + 1) * batch_size)
                local_translator_emb = noise_cache[emb_start_point:emb_end_point]

            # rehearsal data
            if args.generations_only or task_id > 0:
                with torch.no_grad():
                    generations, classes, random_noise, translator_emb = generate_images(args=args,
                                                                                         generator=decoder,
                                                                                         n_prev_examples=n_prev_examples,
                                                                                         task_id=task_id)
                    translator_emb = translator_emb.detach()
                    classes = classes.long()

            # concat local and generated data
            if args.generations_only:
                emb_combined = translator_emb
                classes_combined = classes
            elif task_id > 0:
                emb_combined = torch.cat([translator_emb, local_translator_emb])
                classes_combined = torch.cat([classes, local_classes])
            else:
                emb_combined = local_translator_emb
                classes_combined = local_classes

            # shuffle
            n_mini_batches = math.ceil(len(emb_combined) / batch_size)
            shuffle = torch.randperm(len(emb_combined))
            emb_combined = emb_combined[shuffle]
            classes_combined = classes_combined[shuffle]

            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(classes_combined), (batch_id + 1) * batch_size)

                optimizer.zero_grad()

                do_cutmix = args.cutmix and np.random.rand(1) < args.cutmix_prob
                if do_cutmix:
                    inputs, labels_a, labels_b, lam = cutmix_repr(x=emb_combined[start_point:end_point],
                                                                  y=classes_combined[start_point:end_point],
                                                                  alpha=args.cutmix_alpha)
                    out = classifier(inputs)
                    loss = lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)
                else:
                    out = classifier(emb_combined[start_point:end_point])
                    loss = criterion(out, classes_combined[start_point:end_point].to(classifier.device))

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    acc = get_correct_sum(out, classes_combined[start_point:end_point].to(classifier.device))
                    accuracy += acc.item()
                    total += len(classes_combined[start_point:end_point])
                    losses.append(loss.item())

        scheduler.step()

        if epoch % 1 == 0:
            if args.log_wandb:
                wandb.log({"Classifier loss": np.round(np.mean(losses), 3)})
                wandb.log({"Training accuracy": np.round(accuracy * 100 / total, 3)})
            print("Epoch: {}/{}, loss: {}, Acc: {} %, took: {} s".format(epoch, n_epochs,
                                                                         np.round(np.mean(losses), 3),
                                                                         np.round(accuracy * 100 / total, 3),
                                                                         np.round(time.time() - start), 3))

    return classifier


def get_correct_sum(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum
