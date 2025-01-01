import torch
from tqdm import tqdm
from args import args

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelF1Score, 
    # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html#multilabelf1score
    MultilabelROC, 
    # https://torchmetrics.readthedocs.io/en/stable/classification/roc.html#multilabelroc 
    MultilabelAveragePrecision,
    # https://torchmetrics.readthedocs.io/en/stable/classification/average_precision.html#multilabelaverageprecision
    MultilabelPrecisionRecallCurve
    # https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall_curve.html#multilabelprecisionrecallcurve
)

def validate(encoder, projector, data_loader, transform, metric_fn, device, remove_non_overlapping_classes):

    if remove_non_overlapping_classes:
        NUM_CLASSES = 36
    else:
        NUM_CLASSES = 42 #36 if non-overlapping classes are removed
    
    encoder.eval()
    projector.eval()

    predictions = []
    targets = []
    indices = []

    freq_species = ['SPHSUR', 'BOABIS', 'BOAFAB', 'LEPPOD']
    common_species = ['PITAZU', 'DENMIN', 'PHYCUV', 'LEPLAT', 'PHYALB', 'SCIPER', 'DENNAN', 'BOAALB']
    rare_species = ['DENNAH', 'RHIICT', 'BOALEP', 'ELABIC', 'BOAPRA', 'DENCRU', 'BOALUN', 'PHYMAR', 'PHYSAU', 'LEPFUS', 'LEPLAB', 'BOARAN', 'SCIFUV', 'AMEPIC', 'ADEDIP', 'ELAMAT', 'PHYNAT', 'LEPNOT', 'ADEMAR', 'BOAALM', 'PHYDIS', 'RHIORN', 'DENELE', 'SCIALT', 'SCINAS', 'SCIRIZ', 'LEPELE', 'RHISCI', 'LEPFLA', 'SCIFUS']

    if remove_non_overlapping_classes:
        rare_species = [ele for ele in rare_species if ele not in['LEPELE', 'RHISCI', 'SCINAS', 'LEPFLA', 'SCIRIZ', 'SCIFUS']]

    metric_collection = MetricCollection([
            MultilabelF1Score(num_labels=NUM_CLASSES, average=None, threshold=0.5).to(args.device),
            MultilabelAveragePrecision(num_labels=NUM_CLASSES, average=None, thresholds=None).to(args.device),
            MultilabelROC(num_labels=NUM_CLASSES, thresholds=None).to(args.device),
            MultilabelPrecisionRecallCurve(num_labels=NUM_CLASSES, thresholds=None).to(args.device)
        ])

    for batch_idx, (input, target, index) in enumerate(tqdm(data_loader)):
        input, target = input.to(device), target.to(device)

        with torch.no_grad():

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                prediction = projector(encoder(transform(input)))
                predictions.append(prediction)
                targets.append(target)
                indices.append(index)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    indices = torch.cat(indices)
    multif1 = metric_fn(predictions, targets)

    result_ev = calculate_metrics(predictions, targets, metric_collection)
    multif1 = result_ev['MultilabelF1Score']

    if not remove_non_overlapping_classes:

        class_mapping = [
            'SPHSUR', 'BOABIS', 'SCIPER', 'DENNAH', 'LEPLAT', 'RHIICT', 'BOALEP',
            'BOAFAB', 'PHYCUV', 'DENMIN', 'ELABIC', 'BOAPRA', 'DENCRU', 'BOALUN',
            'BOAALB', 'PHYMAR', 'PITAZU', 'PHYSAU', 'LEPFUS', 'DENNAN', 'PHYALB',
            'LEPLAB', 'SCIFUS', 'BOARAN', 'SCIFUV', 'AMEPIC', 'LEPPOD', 'ADEDIP',
            'ELAMAT', 'PHYNAT', 'LEPELE', 'RHISCI', 'SCINAS', 'LEPNOT', 'ADEMAR',
            'BOAALM', 'PHYDIS', 'RHIORN', 'LEPFLA', 'SCIRIZ', 'DENELE', 'SCIALT'
        ]

    else:

        class_mapping = [
            'SPHSUR', 'BOABIS', 'SCIPER', 'DENNAH', 'LEPLAT', 'RHIICT', 'BOALEP',
            'BOAFAB', 'PHYCUV', 'DENMIN', 'ELABIC', 'BOAPRA', 'DENCRU', 'BOALUN',
            'BOAALB', 'PHYMAR', 'PITAZU', 'PHYSAU', 'LEPFUS', 'DENNAN', 'PHYALB',
            'LEPLAB', 'BOARAN', 'SCIFUV', 'AMEPIC', 'LEPPOD', 'ADEDIP',
            'ELAMAT', 'PHYNAT', 'LEPNOT', 'ADEMAR',
            'BOAALM', 'PHYDIS', 'RHIORN', 'DENELE', 'SCIALT'
        ]

    multif1_freq = []
    multif1_common = []
    multif1_rare = []

    for i, cls in enumerate(class_mapping):

        if cls in freq_species:
            multif1_freq.append(multif1[i])
        elif cls in common_species:
            multif1_common.append(multif1[i])
        elif cls in rare_species:
            multif1_rare.append(multif1[i])

    multif1_freq = torch.tensor(multif1_freq).mean().item()
    multif1_common = torch.tensor(multif1_common).mean().item()
    multif1_rare = torch.tensor(multif1_rare).mean().item()

    return multif1.mean().item(), multif1_freq, multif1_common, multif1_rare

def calculate_metrics(preds, targets, fn_metrics):
    return fn_metrics(preds, targets.long())