import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.utils import logits_to_probs

from vae_experiments.lap_loss import LapLoss
from vae_experiments.latent_visualise import Visualizer
from vae_experiments.vae_utils import *
import copy
import wandb

torch.autograd.set_detect_anomaly(True)


def entropy(logits):
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def loss_fn(y, x_target, mu, sigma, marginal_loss, scale_marginal_loss=1, lap_loss_fn=None):
    marginal_likelihood = scale_marginal_loss * marginal_loss(y, x_target) / y.size(0)
    KL_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / y.size(0)
    if lap_loss_fn:
        lap_loss = lap_loss_fn(y, x_target)
        loss = marginal_likelihood + scale_marginal_loss* 10 * x_target[0].size()[1] * x_target[0].size()[1] * lap_loss + KL_divergence
    else:
        loss = marginal_likelihood + KL_divergence

    return loss, KL_divergence



def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)




def train_local_generator(local_vae, dataset, task_loader, task_id, n_classes, n_epochs=100, scale_marginal_loss=1,
                          use_lap_loss=False, local_start_lr=0.001, scheduler_rate=0.99, scale_local_lr=False):
    # n_epochs = 1
    local_vae.train()
    local_vae.decoder.translator.train()
    translate_noise = True
    starting_point = task_id
    print(f"Selected {starting_point} as staring point for task {task_id}")
    local_vae.starting_point = starting_point
    # if scale_local_lr:
    # lr = min_loss * local_start_lr
    # else:
    assert (not scale_local_lr)
    lr = local_start_lr
    print(f"lr set to: {lr}")
    table_tmp = torch.zeros(n_classes, dtype=torch.long)
    lap_loss = LapLoss(device=local_vae.device) if use_lap_loss else None
    if dataset == "MNIST":
        marginal_loss = nn.BCELoss(reduction="sum")
    else:
        marginal_loss = nn.MSELoss(reduction="sum")

    if task_id > 0:
        optimizer = torch.optim.Adam(list(local_vae.encoder.parameters()), lr=lr / 10, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    else:
        optimizer = torch.optim.Adam(list(local_vae.parameters()), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(n_epochs):
        losses = []
        kl_divs = []
        bin_kl_divs = [0]
        start = time.time()

        if (task_id != 0) and (epoch == min(20, max(n_epochs // 10, 5))):
            print("End of local_vae pretraining")
            optimizer = torch.optim.Adam(list(local_vae.parameters()), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
        gumbel_temp = max(1 - (5 * epoch / (n_epochs)), 0.01)
        if gumbel_temp < 0.1:
            gumbel_temp = None
        
        if epoch == n_epochs - 1:
            ones_distribution = torch.zeros([n_classes, local_vae.binary_latent_size]).to(local_vae.device)

        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(local_vae.device)
            y = batch[1]

            starting_point = y # class dependant

            recon_x, mean, log_var, z, bin_x, binary_prob = local_vae(x, starting_point, y, temp=gumbel_temp,
                                                                      translate_noise=translate_noise)

            loss, kl_div = loss_fn(recon_x, x, mean, log_var, marginal_loss, scale_marginal_loss, lap_loss)
            loss_final = loss  # + binary_loss
            optimizer.zero_grad()
            loss_final.backward()
            nn.utils.clip_grad_value_(local_vae.parameters(), 4.0)
            optimizer.step()

            kl_divs.append(kl_div.item())
            losses.append(loss.item())

            if epoch == 0:
                class_counter = torch.unique(y, return_counts=True)
                table_tmp[class_counter[0]] += class_counter[1].cpu()
            
            if epoch == n_epochs - 1:
                # distinguish ones distribution between two classes
                bin_x = (bin_x / 2 + 0.5)
                for i in range(len(y)):
                    ones_distribution[y[i]] += bin_x[i]


        scheduler.step()
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, kl_div: {},bin_kl: {}, took: {} s".format(epoch, n_epochs,
                                                                                     np.round(np.mean(losses), 3),
                                                                                     np.round(np.mean(kl_divs), 3),
                                                                                     np.round(np.mean(bin_kl_divs), 3),
                                                                                     np.round(time.time() - start), 3))


    # Get the mean distribution
    for i, distribution in enumerate(ones_distribution):
        ones_distribution[i] = ones_distribution[i] / max(1, table_tmp[i])

    if local_vae.decoder.ones_distribution == None:
        # local_vae.decoder.ones_distribution = (ones_distribution.cpu().detach()).view(1, -1)
        local_vae.decoder.ones_distribution = (ones_distribution.cpu().detach()).unsqueeze(dim=0)
        print(f'local vae ones_distribution: {local_vae.decoder.ones_distribution}')

    else:
        # local_vae.decoder.ones_distribution = torch.cat([local_vae.decoder.ones_distribution, (ones_distribution.cpu().detach()).view(1, -1)], 0)
        local_vae.decoder.ones_distribution = torch.cat((local_vae.decoder.ones_distribution, (ones_distribution.cpu().detach()).unsqueeze(dim=0)), 0)
        print(f'local vae ones_distribution: {local_vae.decoder.ones_distribution}')

    return table_tmp


def train_global_decoder(curr_global_decoder, local_vae, task_id, class_table,
                         models_definition, dataset, cosine_sim, n_epochs=100, n_iterations=30, batch_size=1000,
                         train_same_z=False,
                         new_global_decoder=False, global_lr=0.0001, scheduler_rate=0.99, limit_previous_examples=1,
                         warmup_rounds=20,
                         train_loader=None,
                         train_dataset_loader_big=None, num_current_to_compare=1000, experiment_name=None,
                         visualise_latent=False):
    if new_global_decoder:
        global_decoder = models_definition.Decoder(latent_size=curr_global_decoder.latent_size, d=curr_global_decoder.d,
                                                   p_coding=curr_global_decoder.p_coding,
                                                   n_dim_coding=curr_global_decoder.n_dim_coding,
                                                   cond_p_coding=curr_global_decoder.cond_p_coding,
                                                   cond_n_dim_coding=curr_global_decoder.cond_n_dim_coding,
                                                   cond_dim=curr_global_decoder.cond_dim,
                                                   device=curr_global_decoder.device,
                                                   translator=models_definition.Translator(
                                                       curr_global_decoder.n_dim_coding, curr_global_decoder.p_coding,
                                                       curr_global_decoder.latent_size, curr_global_decoder.device),
                                                   standard_embeddings=curr_global_decoder.standard_embeddings,
                                                   trainable_embeddings=curr_global_decoder.trainable_embeddings,
                                                   in_size=curr_global_decoder.in_size
                                                   ).to(curr_global_decoder.device)
    else:
        global_decoder = copy.deepcopy(curr_global_decoder)
    curr_global_decoder.eval()
    curr_global_decoder.translator.eval()
    local_vae.eval()
    global_decoder.train()
    global_decoder.ones_distribution = local_vae.decoder.ones_distribution
    curr_global_decoder.ones_distribution = local_vae.decoder.ones_distribution


    optimizer = torch.optim.Adam(list(global_decoder.translator.parameters()), lr=global_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    if dataset == "MNIST":
        criterion = nn.BCELoss(reduction="sum")
    else:
        criterion = nn.MSELoss(reduction="sum")

    n_prev_examples = int(batch_size * min(task_id, 3) * limit_previous_examples)

    tmp_decoder = curr_global_decoder
    noise_diff_threshold = cosine_sim

    if visualise_latent:
        visualizer = Visualizer(global_decoder, class_table, task_id=task_id, experiment_name=experiment_name)
    
    # n_epochs = 1
    for epoch in range(n_epochs):
        losses = []
        start = time.time()
        sum_changed = torch.zeros([task_id + 1])
        if visualise_latent or (epoch >= warmup_rounds):
            orig_images = next(iter(train_dataset_loader_big))
            means, log_var, bin_z = local_vae.encoder(orig_images[0].to(local_vae.device),
                                                      orig_images[1].to(local_vae.device))
            std = torch.exp(0.5 * log_var)
            binary_out = torch.distributions.Bernoulli(logits=bin_z).sample()
            z_bin_current_compare = binary_out * 2 - 1
            eps = torch.randn([len(orig_images[0]), local_vae.latent_size]).to(local_vae.device)
            z_current_compare = eps * std + means

            # task_ids_current_compare = torch.zeros(len(orig_images[0])) + task_id
            task_ids_current_compare = torch.zeros(len(orig_images[0])) + orig_images[1]

        if visualise_latent:
            visualizer.visualize_latent(local_vae.encoder, global_decoder, epoch, experiment_name,
                                        orig_images=orig_images[0], orig_labels=orig_images[1])
        if epoch == warmup_rounds:
            optimizer = torch.optim.Adam(list(global_decoder.parameters()), lr=global_lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)

        for iteration in range(n_iterations):
            # Building dataset from previous global model and local model
            recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(
                tmp_decoder,
                class_table=class_table,
                n_tasks=task_id,
                n_img=n_prev_examples,
                num_local=batch_size,
                return_z=True,
                translate_noise=True,
                same_z=train_same_z,
                equal_split=True)

            if train_same_z:
                z_prev, z_max, z_bin_prev, z_bin_max = z_prev
            else:
                z_prev, z_bin_prev = z_prev

            with torch.no_grad():
                recon_local, sampled_classes_local, _ = next(iter(train_loader))

                # task_ids_local = torch.zeros([len(recon_local)]) + task_id
                task_ids_local = torch.zeros([len(recon_local)]) + sampled_classes_local
                
                recon_local = recon_local.to(local_vae.device)
                means, log_var, bin_z = local_vae.encoder(recon_local, sampled_classes_local)
                std = torch.exp(0.5 * log_var)
                binary_out = torch.distributions.Bernoulli(logits=bin_z).sample()
                z_bin_local = binary_out * 2 - 1
                eps = torch.randn([len(recon_local), local_vae.latent_size]).to(local_vae.device)
                z_local = eps * std + means

            z_concat = torch.cat([z_prev, z_local])
            z_bin_concat = torch.cat([z_bin_prev, z_bin_local])

            # task_ids_concat = torch.cat([task_ids_prev, task_ids_local])
            task_ids_concat = torch.cat([classes_prev, task_ids_local])

            class_concat = torch.cat([classes_prev, sampled_classes_local.view(-1)])
            if epoch > warmup_rounds:  # Warm-up epochs within which we don't switch targets
                with torch.no_grad():
                    current_noise_translated = global_decoder.translator(z_current_compare, z_bin_current_compare,
                                                                         task_ids_current_compare)

                    # prev_noise_translated = global_decoder.translator(z_prev, z_bin_prev, task_ids_prev)
                    prev_noise_translated = global_decoder.translator(z_prev, z_bin_prev, classes_prev)
                    
                    noise_simmilairty = 1 - cosine_distance(prev_noise_translated,
                                                            current_noise_translated)
                    selected_examples = torch.max(noise_simmilairty, 1)[0] > noise_diff_threshold
                    if selected_examples.sum() > 0:
                        selected_replacements = torch.max(noise_simmilairty, 1)[1][selected_examples]
                        selected_new_generations = orig_images[0][selected_replacements].to(global_decoder.device)
                        recon_prev[selected_examples] = selected_new_generations

                    # switches = torch.unique(task_ids_prev[selected_examples], return_counts=True)
                    switches = torch.unique(classes_prev[selected_examples], return_counts=True)

                    for prev_task_id, sum in zip(switches[0], switches[1]):
                        pass
                        # sum_changed[int(prev_task_id.item())] += sum

            recon_concat = torch.cat([recon_prev, recon_local])

            n_mini_batches = math.ceil(len(z_concat) / batch_size)
            shuffle = torch.randperm(len(task_ids_concat))
            z_concat = z_concat[shuffle]
            z_bin_concat = z_bin_concat[shuffle]
            task_ids_concat = task_ids_concat[shuffle]
            recon_concat = recon_concat[shuffle]
            class_concat = class_concat[shuffle]

            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(task_ids_concat), (batch_id + 1) * batch_size)
                global_recon = global_decoder(z_concat[start_point:end_point], z_bin_concat[start_point:end_point],
                                            #   task_ids_concat[start_point:end_point],
                                              class_concat[start_point:end_point],
                                              class_concat[start_point:end_point])
                loss = criterion(global_recon, recon_concat[start_point:end_point])
                global_decoder.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
        scheduler.step()
        if (epoch % 1 == 0):
            print("Epoch: {}/{}, loss: {}, took: {} s".format(epoch, n_epochs, np.mean(losses), time.time() - start))
            if sum_changed.sum() > 0:
                print(
                    f"Epoch: {epoch} - changing from batches: {[(idx, n_changes) for idx, n_changes in enumerate(sum_changed.tolist())]}")
    return global_decoder

def train_feature_extractor(feature_extractor, task_loader, n_epochs,
                            local_start_lr=0.001, scheduler_rate=0.99):
    # n_epochs = 1
    feature_extractor.train()
    lr = local_start_lr
    print(f"feature extractor's lr set to: {lr}")

    criterion = nn.MSELoss(reduction="sum") # add criterion
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()), lr=lr / 10, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    
    for epoch in range(n_epochs): # parametrise f-e epochs
        losses = []
        cosine_distances = []
        start = time.time()

        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(feature_extractor.device)
            y = batch[1]

            optimizer.zero_grad()

            out = feature_extractor(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # cosine similiarity measure
            with torch.no_grad():
                for i, output in enumerate(out):
                    cosine_distances.append((torch.cosine_similarity(output, y[i], dim=0)).item())

        scheduler.step()
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, cosine similarity: {}, took: {} s".format(epoch, n_epochs,
                                                                                np.round(np.mean(losses), 3),
                                                                                np.round(np.mean(cosine_distances), 3),
                                                                                np.round(time.time() - start), 3))

    return feature_extractor

def train_binary_head(head, task_loader, fe, n_epochs,
                        local_start_lr=0.001, scheduler_rate=0.99):
    head.train()
    lr = local_start_lr
    print(f"head's lr set to: {lr}")

    optimizer = torch.optim.Adam(list(head.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        losses = []
        accuracy = 0
        total = 0
        start = time.time()

        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(head.device)
            y = batch[1].to(head.device)

            optimizer.zero_grad()

            with torch.no_grad():
                extracted = fe(x)

            out = head(extracted)
            out = out.squeeze(1)
            loss = criterion(out, y)
            acc = get_binary_accuracy(out, y)
            
            loss.backward()
            optimizer.step()

            accuracy += acc.item()
            total += y.shape[0]
            losses.append(loss.item())

        scheduler.step()
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, Acc: {} %, took: {} s".format(epoch, n_epochs,
                                                                np.round(np.mean(losses), 3),
                                                                np.round(accuracy / total, 3),
                                                                np.round(time.time() - start), 3))

    return head

def train_head(head, task_loader, fe, n_epochs, local_start_lr=0.001, scheduler_rate=0.99):
    # n_epochs = 3
    wandb.watch(head)
    head.train()
    lr = local_start_lr
    print(f"head's lr set to: {lr}")

    optimizer = torch.optim.Adam(list(head.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        losses = []
        accuracy = 0
        total = 0
        start = time.time()

        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(head.device)
            y = batch[1].to(head.device)

            optimizer.zero_grad()

            with torch.no_grad():
                extracted = fe(x)

            out = head(extracted)
            out = out.squeeze(1)
            loss = criterion(out, y)
            acc = get_head_accuracy(out, y)
            
            loss.backward()
            optimizer.step()

            accuracy += acc.item()
            total += len(y)
            losses.append(loss.item())
            wandb.log({"training_loss": (loss.item())})

        scheduler.step()
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, Acc: {} %, took: {} s".format(epoch, n_epochs,
                                                                np.round(np.mean(losses), 3),
                                                                np.round(accuracy * 100 / total, 3),
                                                                np.round(time.time() - start), 3))

    return head


def get_head_accuracy(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum

def get_binary_accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    
    return correct_results_sum

# def get_accuracy(model, fe, data, device):
#     correct = 0
#     total = 0
#     model.eval()
#     fe.eval()
#     with torch.no_grad():
#         for iteration, batch in enumerate(data):
            
#             imgs, labels = batch[0].to(device), batch[1].to(device)

#             with torch.no_grad():
#                 extracted = fe(imgs)
            
#             output = model(extracted)

#             pred = output.max(1, keepdim=True)[1] # get the index of the max logit
#             correct += pred.eq(labels).sum().item()
#             total += imgs.shape[0]
#     return correct*100 / total