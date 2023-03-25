import torch
from vae_experiments import training_functions


def train_classifier(args, feature_extractor, classifier, train_loader, task_id, device):

    # Load the generator
    if args.generator_type == "vae":
        generator_path = f'results/vae/{args.experiment_name}/model{task_id}_curr_decoder'
        generator = torch.load(generator_path).to(device)
    elif args.generator_type == "gan":
        generator_path = f'results/gan/{args.experiment_name}/model{task_id}_curr_global_generator'
        generator = torch.load(generator_path, map_location="cuda")
        generator = generator.to(device)
    else:
        print(f'Unknown generator type: {args.generator_type}')
        raise NotImplementedError
        
    print(f'Loaded generator')
    print(generator)

    # Classifier training
    if args.gen_load_feature_extractor:
        feature_extractor = torch.load(f'results/{args.generator_type}/{args.experiment_name}/model{99}_feature_extractor').to(device)
        print("Loaded feature extractor")
    else:
        print("\nTrain feature extractor")
        feature_extractor = training_functions.train_feature_extractor(args=args,
                                                                       feature_extractor=feature_extractor,
                                                                       train_loader=None,
                                                                       n_epochs=args.feature_extractor_epochs,
                                                                       decoder=generator,
                                                                       task_id=task_id,
                                                                       batch_size=args.gen_batch_size)
        print("Done training feature extractor\n")


    if args.gen_load_classifier:
        classifier = torch.load(f'results/{args.generator_type}/{args.experiment_name}/model{99}_classifier').to(device)
        print("Loaded head")
    else:
        print("\nTrain classifier head")
        classifier = training_functions.train_head(args=args,
                                                    classifier=classifier, 
                                                    n_epochs=args.classifier_epochs,
                                                    decoder=generator,
                                                    task_id=task_id,
                                                    batch_size=args.gen_batch_size)
        print("Done training classifier head\n")

    torch.cuda.empty_cache()
    return feature_extractor, classifier
