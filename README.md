# Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds
Authors: Ilyass Moummad, Nicolas Farrugia, Romain Serizel, Jeremy Froidevaux, Vincent Lostanlen
---

### Preprint: https://arxiv.org/abs/2403.09598

We introduce a framework that leverages mixing regularization methods Mixup, Manifold Mixup, and MultiMix to handle multi-label and class imbalance on the [Anuraset](https://github.com/soundclim/anuraset) dataset.

We base our code on the official implementation of AnuraSet baseline: https://github.com/soundclim/anuraset where you can find the link to download the dataset [AnuraSet](https://www.nature.com/articles/s41597-023-02666-2) \\
Python libraries required : torch, torchmetrics, numpy, pandas, tqdm

```main.py```: main code for training and evaluating on AnuraSet \
```dataset.py```: dataset class \
```models.py```: code for MobileNetV3 model \
```train.py```: train utility functions \
```val.py```: evaluation utility functions \
```transforms.py```: transformation classes \
```args.py```: argparse of the arguments \

### Training and Evaluation
python3 main.py --rootdir dataset_path --mix mix2 --device 'cuda' --sr 16000 --workers 16 --save

## To cite this work:
```
@misc{2403.09598,
Author = {Ilyass Moummad and Nicolas Farrugia and Romain Serizel and Jeremy Froidevaux and Vincent Lostanlen},
Title = {Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds},
Year = {2024},
Eprint = {arXiv:2403.09598},
}
```