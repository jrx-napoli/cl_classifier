from base64 import encode
import torch
import numpy as np
import random
from vae_experiments import vae_utils
import torch.utils.data as data
import matplotlib.pyplot as plt

class FeatureExtractorDataset(data.Dataset):
    def __init__(self, datasets, encoder, translator, decoder, task_id, class_table, latent_size, gen_batch_size):
        
        batch_size = gen_batch_size
        task_id_int_copy = task_id
        if not torch.is_tensor(task_id):
            if task_id != None:
                task_id = torch.zeros([batch_size, 1]) + task_id
            else:
                task_id = torch.zeros([batch_size, 1])

        img, labels = self._create_dataset(datasets=datasets, 
                                            encoder=encoder, 
                                            translator=translator, 
                                            decoder=decoder, 
                                            task_id=task_id, 
                                            n_tasks=task_id_int_copy, 
                                            class_table=class_table, 
                                            latent_size=latent_size,
                                            gen_batch_size=gen_batch_size,
                                            gen_samples_only=True)
        self.img_data = img
        self.img_labels = labels
        self.dataset_len = len(self.img_labels)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.img_data[idx], self.img_labels[idx]

    def _create_dataset(self, datasets, encoder, translator, decoder, task_id, n_tasks, class_table, latent_size, gen_batch_size, gen_samples_only=True):
        x = []
        y = []
        num_samples = 0
        if datasets != None:
            for dataset in datasets:
                num_samples += len(dataset)

        print(f'\nCreating FeatureExtractorDataset')
        print(f'-> number of provided samples: {num_samples}')
        print(f'-> gen_samples_only: {gen_samples_only}')
        
        with torch.no_grad():
            batch_size = gen_batch_size
            
            # Real samples from current task
            if not gen_samples_only:
                print(f'Adding real samples...')
                for dataset in datasets:
                    for sample, target in iter(dataset):
                        x.append(sample.to(encoder.device))
                        means, log_var = encoder(sample.to(encoder.device), target)
                        std = torch.exp(0.5 * log_var)
                        batch_size = sample.size(0)
                        eps = torch.randn([batch_size, latent_size]).to(encoder.device)
                        z = eps * std + means
                        new_data = translator(z, task_id)
                        y.append(torch.squeeze(new_data))

            # Generated-only images from current task
            n_prev_examples = 6000 + (min(n_tasks, 2) * 1000)
            recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = vae_utils.generate_previous_data(
                decoder,
                class_table=class_table,
                n_tasks=n_tasks+1,
                n_img=n_prev_examples,
                num_local=batch_size,
                return_z=True,
                translate_noise=True,
                same_z=False,
                equal_split=True,
                recent_task_only=False)

            # add generated samples to dataset
            current_task_counter = 0
            for i in range(n_prev_examples):
                current_task_id = torch.zeros([1, 1]) + classes_prev[i]
                translated_z = translator(torch.unsqueeze(z_prev[i], 0), current_task_id)
                x.append(recon_prev[i])
                y.append(torch.squeeze(translated_z))
                current_task_counter += 1
            print(f'Adding {current_task_counter} generated samples from current task...')

            # generated samples from previous tasks
            if -1 > 0:
                n_prev_examples = current_task_counter
                recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = vae_utils.generate_previous_data(
                    decoder,
                    class_table=class_table,
                    n_tasks=n_tasks,
                    n_img=n_prev_examples,
                    num_local=batch_size,
                    return_z=True,
                    translate_noise=True,
                    same_z=False,
                    equal_split=True)

                # add generated samples to dataset
                for i in range(n_prev_examples):
                    current_task_id = torch.zeros([1, 1]) + classes_prev[i]
                    translated_z = translator(torch.unsqueeze(z_prev[i], 0), current_task_id)
                    x.append(recon_prev[i])
                    y.append(torch.squeeze(translated_z))
                print(f'Adding {n_prev_examples} generated samples from previous tasks...')
                
        print(f'Done creating FeatureExtractorDataset\n')
        return x, y

class HeadDataset(data.Dataset):
    def __init__(self, datasets, encoder, decoder, task_id, class_table, binary_head, gen_samples_only, gen_batch_size, global_benchmark=False):
        
        batch_size = gen_batch_size
        task_id_int_copy = task_id
        if not torch.is_tensor(task_id):
            if task_id != None:
                task_id = torch.zeros([batch_size, 1]) + task_id
            else:
                task_id = torch.zeros([batch_size, 1])

        img, labels = self._create_dataset(datasets=datasets, 
                                            encoder=encoder, 
                                            decoder=decoder, 
                                            task_id=task_id, 
                                            n_tasks=task_id_int_copy, 
                                            class_table=class_table, 
                                            gen_samples_only = gen_samples_only,
                                            gen_batch_size=gen_batch_size,
                                            global_benchmark=global_benchmark)

        self.img_data = img
        self.img_labels = labels
        self.dataset_len = len(self.img_labels)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.img_data[idx], self.img_labels[idx]


    def _create_dataset(self, datasets, encoder, decoder, task_id, n_tasks, class_table, gen_batch_size, 
                        gen_samples_only=True, global_benchmark=False):
                        
        x = []
        y = []
        num_samples = 0
        if datasets != None:
            for dataset in datasets:
                num_samples += len(dataset)

        print(f'\nCreating HeadDataset')
        print(f'-> number of provided samples: {num_samples}')
        print(f'-> gen_samples_only: {gen_samples_only}')
        print(f'-> global_benchmark: {global_benchmark}')

        with torch.no_grad():

            # Global benchmark
            if global_benchmark and task_id == 4:
                print('Creating dataset for a global benchmark...')
                batch_size = gen_batch_size
                n_prev_examples = num_samples * 3
                recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = vae_utils.generate_previous_data(
                    decoder,
                    class_table=class_table,
                    n_tasks=5,
                    n_img=n_prev_examples,
                    num_local=batch_size,
                    return_z=True,
                    translate_noise=True,
                    same_z=False,
                    equal_split=True)

                # classify and add generated samples to dataset
                for i in range(n_prev_examples):
                    x.append(recon_prev[i])
                    y.append(classes_prev[i])

            else:

                # real samples
                if not gen_samples_only:
                    print('Adding real samples...')
                    for dataset in datasets:
                        for sample, target in iter(dataset):
                            x.append(sample.to(encoder.device))
                            y.append(float(target))

                # generated samples from current task
                batch_size = gen_batch_size
                n_prev_examples = 6000 + (min(n_tasks, 2) * 1000)
                recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = vae_utils.generate_previous_data(
                    decoder,
                    class_table=class_table,
                    n_tasks=n_tasks+1,
                    n_img=n_prev_examples,
                    num_local=batch_size,
                    return_z=True,
                    translate_noise=True,
                    same_z=False,
                    equal_split=True,
                    recent_task_only=False)
                
                # classify and add generated samples to dataset
                current_task_counter = 0
                for i in range(n_prev_examples):
                    current_task_counter += 1
                    x.append(recon_prev[i])
                    y.append(classes_prev[i])
                print(f'Adding {current_task_counter} generated samples from current task...')

                # generated samples from previous tasks
                if -1 > 0:
                    n_prev_examples = current_task_counter
                    print(f'Adding {n_prev_examples} generated samples from previous tasks...')
                    recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = vae_utils.generate_previous_data(
                        decoder,
                        class_table=class_table,
                        n_tasks=n_tasks,
                        n_img=n_prev_examples,
                        num_local=batch_size,
                        return_z=True,
                        translate_noise=True,
                        same_z=False,
                        equal_split=True)
                    
                    # classify and add generated samples to dataset
                    for i in range(n_prev_examples):
                        x.append(recon_prev[i])
                        y.append(classes_prev[i])

        t = torch.FloatTensor(y)
        print(torch.unique(t, return_counts=True))

        print(f'Done creating HeadDataset\n')
        return x, y

class ClassifierValidator:
    def __init__(self) -> None:
        pass

    def validate_feature_extractor(self, dataset, encoder, translator, decoder, task_id, class_table, latent_size, gen_batch_size, feature_extractor):
        with torch.no_grad():

            cosine_distances = []
            fe_dataset = FeatureExtractorDataset(datasets=None,
                                                    # datasets=[dataset] 
                                                    encoder=encoder, 
                                                    translator=translator, 
                                                    decoder=decoder,
                                                    task_id=task_id, 
                                                    class_table=class_table,
                                                    latent_size=latent_size,
                                                    gen_batch_size=gen_batch_size)
            task_loader = data.DataLoader(dataset=fe_dataset, batch_size=gen_batch_size, shuffle=True, drop_last=False)
            
            for iteration, batch in enumerate(task_loader):
                x = batch[0].to(feature_extractor.device)
                y = batch[1]
                out = feature_extractor(x)
                for i, output in enumerate(out):
                    cosine_distances.append((torch.cosine_similarity(output, y[i], dim=0)).item())
            
            return np.round(np.mean(cosine_distances), 3)
    
    def validate_classifier(self, fe, head, data_loader, binary_head):
        total = 0
        correct = 0

        with torch.no_grad():
            for iteration, batch in enumerate(data_loader):

                x = batch[0].to(head.device)
                y = batch[1].to(head.device)

                extracted = fe(x)
                out = head(extracted)
                out = out.squeeze(1)
                correct_sum = self.get_correct_sum(out, y)
                
                correct += correct_sum.item()
                total += y.shape[0]

        return correct, total

    def validate_global_benchamrk(self, test_model, data_loader):
        total = 0
        correct = 0

        with torch.no_grad():
            for iteration, batch in enumerate(data_loader):

                x = batch[0].to("cuda")
                y = batch[1].to("cuda")

                out = test_model(x)
                correct_sum = self.get_correct_sum(out, y)
                
                correct += correct_sum.item()
                total += y.shape[0]

        return correct, total

    def get_correct_sum(self, y_pred, y_test):
        _, y_pred_tag = torch.max(y_pred, 1)
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        return correct_results_sum
