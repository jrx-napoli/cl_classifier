import torch
import training


def train_classifier(args, feature_extractor, classifier, train_loader, task_id, device):

    # Load the generator
    if args.generator_type == "vae":
        generator_path = f'multiband_vae/results/vae/{args.experiment_name}/model{task_id}_curr_decoder'
        generator = torch.load(generator_path).to(device)
    elif args.generator_type == "gan":
        generator_path = f'multiband_vae/results/gan/{args.experiment_name}/model{task_id}_curr_global_generator'
        generator = torch.load(generator_path, map_location="cuda")
        generator = generator.to(device)
    else:
        print(f'Unknown generator type: {args.generator_type}')
        raise NotImplementedError
        
    print(f'Loaded generator')
    print(generator)

    # Classifier training
    if args.gen_load_feature_extractor:
        feature_extractor = torch.load(f'models/{args.generator_type}/{args.experiment_name}/model{task_id}_feature_extractor').to(device)
        print("Loaded feature extractor")
    else:
        print("\nTrain feature extractor")
        local_translator_emb_cache = torch.load(f"models/{args.generator_type}/{args.experiment_name}/local_translator_emb_cache")
        feature_extractor, local_translator_emb_cache = training.train_feature_extractor(args=args,
                                                                                                    feature_extractor=feature_extractor,
                                                                                                    train_loader=train_loader,
                                                                                                    local_translator_emb_cache=local_translator_emb_cache,
                                                                                                    decoder=generator,
                                                                                                    task_id=task_id,
                                                                                                    device=device)
        print("Done training feature extractor\n")
        torch.save(feature_extractor, f"models/{args.generator_type}/{args.experiment_name}/model{task_id}_feature_extractor")
        # torch.save(local_translator_emb_cache, f"results/{args.generator_type}/{args.experiment_name}/local_translator_emb_cache_BETA")


    if args.gen_load_classifier:
        classifier = torch.load(f'models/{args.generator_type}/{args.experiment_name}/model{task_id}_classifier').to(device)
        print("Loaded head")
    else:
        print("\nTrain classifier head")
        local_translator_emb_cache = torch.load(f"models/{args.generator_type}/{args.experiment_name}/local_translator_emb_cache")
        classifier = training.train_head(args=args,
                                        local_translator_emb_cache=local_translator_emb_cache,
                                        train_loader=train_loader,
                                        classifier=classifier, 
                                        decoder=generator,
                                        task_id=task_id,
                                        device=device)
        print("Done training classifier head\n")

    torch.cuda.empty_cache()
    return feature_extractor, classifier