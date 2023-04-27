import math
import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim

from multiband_vae.vae_experiments import vae_utils
from multiband_vae.gan_experiments import gan_utils
from regularization.cutmix import cutmix_images, cutmix_repr
import wandb

torch.autograd.set_detect_anomaly(True)


def get_fe_criterion(args):
    if args.mse_reduction:
        return nn.MSELoss(reduction="sum")
    else:
        return nn.MSELoss()


def get_optimiser(args, model):
    if args.optimizer.lower() == "adam":
        return torch.optim.Adam(list(model.parameters()), lr=0.001)
    elif args.optimizer.lower() == "sgd":
        return torch.optim.SGD(list(model.parameters()), lr=args.max_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


def get_scheduler(args, optimizer):
    if args.optimizer.lower() == "adam":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.optimizer.lower() == "sgd":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=args.min_lr)
    else:
        raise NotImplementedError


def calculate_gan_noise(args, generator, train_loader, noise_cache, task_id, device):
    print("\nNoise optimisation for GAN embeddings")
    n_iterations = 500

    for iteration, batch in enumerate(train_loader):
        print(f"Batch {iteration}/{n_iterations}")

        local_imgs, local_classes = batch
        local_imgs = local_imgs.to(device)
        local_classes = local_classes.to(device)
        _, local_translator_emb = gan_utils.optimize_noise(images=local_imgs,
                                                           generator=generator,
                                                           n_iterations=n_iterations,
                                                           task_id=task_id,
                                                           lr=0.01,
                                                           labels=local_classes)
        local_translator_emb = local_translator_emb.detach()

        if noise_cache is None:
            noise_cache = local_translator_emb
        else:
            noise_cache = torch.cat((noise_cache, local_translator_emb), 0)

    torch.save(noise_cache, f"models/{args.generator_type}/{args.experiment_name}/model{task_id}_noise_cache")
    return noise_cache


def generate_images(args, batch_size, generator, n_prev_examples, n_tasks):
    if args.generator_type == "vae":
        generations, classes, _, translator_emb = vae_utils.generate_previous_data(
            generator,
            n_tasks=n_tasks,
            n_img=n_prev_examples,
            num_local=batch_size,
            return_z=True,
            translate_noise=True,
            same_z=False)

    elif args.generator_type == "gan":
        # TODO bugfix: number of generations is rounded down
        generations, _, classes, translator_emb = gan_utils.generate_previous_data(
            n_prev_tasks=(5 * n_tasks),  # TODO adjust for specific dataset - n_classes for each tasks?
            n_prev_examples=n_prev_examples,
            curr_global_generator=generator)

    return generations, translator_emb, classes


def train_feature_extractor(args, feature_extractor, decoder, task_id, device, train_loader,
                            noise_cache=None):
    if args.log_wandb:
        wandb.watch(feature_extractor)
    feature_extractor.train()
    decoder.translator.eval()
    decoder.eval()

    n_epochs = args.feature_extractor_epochs
    batch_size = args.batch_size
    n_iterations = len(train_loader)
    # n_prev_examples = int(batch_size * min(task_id, 10))
    n_prev_examples = 95 * 20
    n_tasks = task_id

    print(f'Iterations /epoch: {n_iterations}')
    print(f'Generations /iteration: {n_prev_examples}')
    if args.log_wandb:
        wandb.run.summary["Generations per iteration"] = n_prev_examples

    criterion = get_fe_criterion(args=args)
    optimizer = get_optimiser(args=args, model=feature_extractor)
    scheduler = get_scheduler(args=args, optimizer=optimizer)

    for epoch in range(n_epochs):
        losses = []
        cosine_similarities = []
        start = time.time()

        # Optimise noise and store it for entire training process
        if epoch == 0 and args.generator_type == "gan" and args.calc_noise:
            noise_cache = calculate_gan_noise(args=args, generator=decoder, train_loader=train_loader,
                                              noise_cache=noise_cache, task_id=task_id, device=device)

        for iteration, batch in enumerate(train_loader):

            # local data
            local_imgs, _ = batch
            local_imgs = local_imgs.to(device)
            emb_start_point = iteration * batch_size
            emb_end_point = min(len(train_loader.dataset), (iteration + 1) * batch_size)
            local_translator_emb = noise_cache[emb_start_point:emb_end_point]

            # rehearsal data
            with torch.no_grad():
                generations, translator_emb, _ = generate_images(args=args, batch_size=batch_size, generator=decoder,
                                                                 n_prev_examples=n_prev_examples, n_tasks=n_tasks)
                generations = generations.to(device)

            # concat local and generated data
            generations = torch.cat([generations, local_imgs])
            translator_emb = torch.cat([translator_emb, local_translator_emb])

            # shuffle
            n_mini_batches = math.ceil(len(generations) / batch_size)
            shuffle = torch.randperm(len(generations))
            generations = generations[shuffle]
            translator_emb = translator_emb[shuffle]

            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(generations), (batch_id + 1) * batch_size)

                optimizer.zero_grad()

                do_cutmix = args.cutmix and np.random.rand(1) < args.cutmix_prob
                if do_cutmix:
                    inputs, labels_a, labels_b, lam = cutmix_images(x=generations[start_point:end_point],
                                                                    y=translator_emb[start_point:end_point],
                                                                    alpha=args.cutmix_alpha)

                out = feature_extractor(generations[start_point:end_point])
                loss = lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b) if do_cutmix else criterion(out, translator_emb[start_point:end_point])
                # loss = criterion(out, translator_emb[start_point:end_point])

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    losses.append(loss.item())
                    for i, output in enumerate(out):
                        cosine_similarities.append(
                            (torch.cosine_similarity(output, translator_emb[start_point:end_point][i], dim=0)).item())

        scheduler.step()

        if epoch % 1 == 0:
            if args.log_wandb:
                wandb.log({"Feature-extractor loss": np.round(np.mean(losses), 3)})
                wandb.log({"Mean cosine similarity": np.round(np.mean(cosine_similarities), 3)})
            print("Epoch: {}/{}, loss: {}, cosine similarity: {}, took: {} s".format(epoch,
                                                                                     n_epochs,
                                                                                     np.round(np.mean(losses), 3),
                                                                                     np.round(
                                                                                         np.mean(cosine_similarities),
                                                                                         3),
                                                                                     np.round(time.time() - start), 3))

    return feature_extractor, noise_cache


def train_classifier(args, classifier, decoder, task_id, device,
                     train_loader=None, local_translator_emb_cache=None, scheduler_rate=0.99):
    if args.log_wandb:
        wandb.watch(classifier)
    decoder.translator.eval()
    decoder.eval()
    classifier.train()

    n_epochs = args.classifier_epochs
    batch_size = args.batch_size
    n_iterations = len(train_loader)
    # n_prev_examples = int(batch_size * min(task_id, 10))
    n_prev_examples = 95 * 20
    n_tasks = task_id

    print(f'Iterations /epoch: {n_iterations}')
    print(f'Generations /iteration: {n_prev_examples}')

    optimizer = torch.optim.Adam(list(classifier.parameters()), lr=0.001, weight_decay=1e-5)
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

            # rehearsal data
            _, translator_emb, classes = generate_images(args=args, batch_size=batch_size, generator=decoder,
                                                         n_prev_examples=n_prev_examples, n_tasks=n_tasks)
            classes = classes.long()

            # concat local and generated data
            translator_emb = torch.cat([translator_emb, local_translator_emb])
            classes = torch.cat([classes, local_classes])
            # print(torch.unique(classes, return_counts=True))

            # shuffle
            n_mini_batches = math.ceil(len(classes) / batch_size)
            shuffle = torch.randperm(len(classes))
            translator_emb = translator_emb[shuffle]
            classes = classes[shuffle]

            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(classes), (batch_id + 1) * batch_size)

                optimizer.zero_grad()

                do_cutmix = args.cutmix and np.random.rand(1) < args.cutmix_prob
                if do_cutmix:
                    inputs, labels_a, labels_b, lam = cutmix_repr(x=translator_emb[start_point:end_point],
                                                                  y=classes[start_point:end_point],
                                                                  alpha=args.cutmix_alpha)

                out = classifier(translator_emb[start_point:end_point])
                loss = lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b) if do_cutmix else criterion(out, classes[start_point:end_point])
                # loss = criterion(out, classes[start_point:end_point].to(classifier.device))

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    acc = get_correct_sum(out, classes[start_point:end_point].to(classifier.device))
                    accuracy += acc.item()
                    total += len(classes[start_point:end_point])
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
