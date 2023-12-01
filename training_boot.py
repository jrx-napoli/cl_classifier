import torch
import os

import training


def train_classifier(args, feature_extractor, classifier, train_loader, task_id, device):
    local_vae = None
    noise_cache = None

    # Load generative models
    if args.generator_type == "vae":
        local_vae = torch.load(os.path.join(args.models_root, f"model{task_id}_local_vae")).to(device)
        generator = torch.load(os.path.join(args.models_root, f"model{task_id}_curr_decoder")).to(device)
    elif args.generator_type == "gan":
        generator = torch.load(os.path.join(args.models_root, f"model{task_id}_curr_global_generator")).to(device)
    else:
        raise NotImplementedError
    print(f'Loaded generator')

    # Calculate GAN noise
    if not args.generations_only:
        if args.calc_noise:
            noise_cache = training.calculate_gan_noise(args=args,
                                                    generator=generator,
                                                    train_loader=train_loader,
                                                    task_id=task_id,
                                                    device=device)
        elif args.generator_type == "gan":
            noise_cache = torch.load(os.path.join(args.models_root, f"model{task_id}_noise_cache"))
    else:
        noise_cache = None

    # Train models
    if args.load_feature_extractor:
        feature_extractor = torch.load(os.path.join(args.models_root, f"model{task_id}_feature_extractor")).to(device)
        print("Loaded feature extractor")
    else:
        print("\nTrain feature extractor")
        feature_extractor, noise_cache = training.train_feature_extractor(args=args,
                                                                          feature_extractor=feature_extractor,
                                                                          train_loader=train_loader,
                                                                          noise_cache=noise_cache,
                                                                          decoder=generator,
                                                                          local_vae=local_vae,
                                                                          task_id=task_id,
                                                                          device=device)
        print("Done training feature extractor\n")
        torch.save(feature_extractor, os.path.join(args.models_root, f"model{task_id}_feature_extractor"))
#
    if args.load_classifier:
        classifier = torch.load(os.path.join(args.models_root, f"model{task_id}_classifier")).to(device)
        print("Loaded classifier")
    else:
        print("\nTrain classifier")
        classifier = training.train_classifier(args=args,
                                               classifier=classifier,
                                               train_loader=train_loader,
                                               noise_cache=noise_cache,
                                               decoder=generator,
                                               local_vae=local_vae,
                                               task_id=task_id,
                                               device=device)
        print("Done training classifier\n")
        torch.save(classifier, os.path.join(args.models_root, f"model{task_id}_classifier"))

    torch.cuda.empty_cache()
    return feature_extractor, classifier
