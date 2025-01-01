# Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds

**Authors:** Ilyass Moummad, Nicolas Farrugia, Romain Serizel, Jeremy Froidevaux, Vincent Lostanlen

---

Presented at EUSIPCO 2024 - Special Session on Signal Analysis for Biodiversity. Access the full paper [here](https://arxiv.org/abs/2403.09598).

This project introduces a novel framework utilizing mixing regularization methods—Mixup, Manifold Mixup, and MultiMix—to address challenges in multi-label classification and class imbalance within the Anuraset dataset.

The implementation is based on the official AnuraSet baseline, available on [GitHub](https://github.com/soundclim/anuraset). You can also download the dataset directly from [Nature's publication](https://www.nature.com/articles/s41597-023-02666-2).

## Required Python Libraries

```pip install -r requirements```

## Repository Structure

- **main.py**: Main script for training and evaluation.
- **dataset.py**: Dataset class for handling data operations.
- **models.py**: Contains the MobileNetV3 model implementation.
- **train.py**: Utility functions for model training.
- **val.py**: Utility functions for model validation.
- **transforms.py**: Data transformation classes.
- **args.py**: Argument parsing.

## Training and Evaluation

- **Original AnuraSet**: ```python3 main.py --dataset anuraset --rootdir <dataset_path> --mix mix2 --device cuda --sr 16000 --workers 16 --save```

- **AnuraSet-36N** (removing non-overlapping classes between training and testing): --dataset anuraset36n

- **AnuraSet-36** (removing non-overlapping classes as well as silence examples): --dataset anuraset36

Replace `<dataset_path>` with the actual path to your dataset.

## Citation

If you find this work useful, please cite it:

```bibtex
@misc{2403.09598,
  Author = {Ilyass Moummad and Nicolas Farrugia and Romain Serizel and Jeremy Froidevaux and Vincent Lostanlen},
  Title = {Mixture of Mixups for Multi-label Classification of Rare Anuran Sounds},
  Year = {2024},
  Eprint = {arXiv:2403.09598},
}