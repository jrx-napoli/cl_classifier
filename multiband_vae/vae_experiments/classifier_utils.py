import torch
import numpy as np
import torch.utils.data as data


class ClassifierValidator:
    def __init__(self) -> None:
        pass

    def validate_feature_extractor(self, feature_extractor, encoder, translator, data_loader):
        cosine_distances = []
        encoder.eval()
        translator.eval()
        feature_extractor.eval()
        
        with torch.no_grad():
    
            for iteration, batch in enumerate(data_loader):
                
                x = batch[0].to(feature_extractor.device)
                y = batch[1]
                
                out = feature_extractor(x)
                reference = translator(encoder(x, y), y)

                for i, output in enumerate(out):
                    cosine_distances.append((torch.cosine_similarity(output, reference, dim=0)).item())
            
            return np.round(np.mean(cosine_distances), 3)
    
    def validate_classifier(self, feature_extractor, classifier, data_loader):
        total = 0
        correct = 0
        feature_extractor.eval()
        classifier.eval()

        with torch.no_grad():
            for iteration, batch in enumerate(data_loader):

                x = batch[0].to(classifier.device)
                y = batch[1].to(classifier.device)

                extracted = feature_extractor(x)
                out = classifier(extracted)
                correct_sum = self.get_correct_sum(out, y)
                
                correct += correct_sum.item()
                total += y.shape[0]

        return correct, total

    def get_correct_sum(self, y_pred, y_test):
        _, y_pred_tag = torch.max(y_pred, 1)
        # print(y_pred_tag)
        # print(y_test)
        # print('+++++++++')
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        return correct_results_sum


def get_global_eval_dataloaders(task_names, val_dataset_splits, args):
    # Eval dataset contains datasets from all previous tasks
    eval_loaders = []
    for task_id in task_names:
        datasets = []

        for i in range(task_id + 1):
            datasets.append(val_dataset_splits[i])

        eval_data = data.ConcatDataset(datasets)
        eval_loader = data.DataLoader(dataset=eval_data, batch_size=args.val_batch_size, shuffle=True, num_workers=args.workers)
        eval_loaders.append(eval_loader)
    
    return eval_loaders