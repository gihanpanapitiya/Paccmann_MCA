[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml)

# paccmann_predictor

Drug interaction prediction with PaccMann.

`paccmann_predictor` is a package for drug interaction prediction, with examples of 
anticancer drug sensitivity prediction and drug target affinity prediction. Please see the paper:

- [_Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders_](https://doi.org/10.1021/acs.molpharmaceut.9b00520) (*Molecular Pharmaceutics*, 2019). This is the original paper on IC50 prediction using drug properties and tissue-specific cell line information (gene expression profiles). While the original code was written in `tensorflow` and is available [here](https://github.com/drugilsberg/paccmann), this is the `pytorch` implementation of the best PaccMann architecture (multiscale convolutional encoder).


*NOTE*: PaccMann acronyms "Prediction of AntiCancer Compound sensitivity with Multi-modal Attention-based Neural Networks".


## Requirements

- `conda>=3.7`

## Installation

This model is curated as a part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE)
The original code is [_here_](https://github.com/PaccMann/paccmann_predictor)


Create a conda environment:

```sh
conda env create -f examples/IC50/conda.yml
```

Activate the environment:

```sh
conda activate paccmann_predictor
```

Install in editable mode for development:

```sh
pip install -e .
```
Install CANDLE package
```sh
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

## Example usage

```sh
python candle_main.py
```

## References

If you use `paccmann_predictor` in your projects, please cite the following:

```bib
@article{manica2019paccmann,
  title={Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders},
  author={Manica, Matteo and Oskooei, Ali and Born, Jannis and Subramanian, Vigneshwari and S{\'a}ez-Rodr{\'\i}guez, Julio and Mart{\'\i}nez, Mar{\'\i}a Rodr{\'\i}guez},
  journal={Molecular pharmaceutics},
  volume={16},
  number={12},
  pages={4797--4806},
  year={2019},
  publisher={ACS Publications},
  doi = {10.1021/acs.molpharmaceut.9b00520},
  note = {PMID: 31618586}
}

@article{born2021datadriven,
  author = {Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a},
  doi = {10.1088/2632-2153/abe808},
  issn = {2632-2153},
  journal = {Machine Learning: Science and Technology},
  number = {2},
  pages = {025024},
  title = {{Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2}},
  url = {https://iopscience.iop.org/article/10.1088/2632-2153/abe808},
  volume = {2},
  year = {2021}
}
```
