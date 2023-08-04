## Introduction
This repository contains the code for group project "Answer Retrieval for Questions on Math Problem Formulation" in the course "Introduction to Natural Language Processing" at the University of Bonn.

The project aims to solve the information retrieval on math dataset is based on the papers [[1]](#1), [[2]](#2), [[3]](#3). We extended the work of [[1]](#1) by altering the preprocessing step for the fine-tune dataset. Therefore we adapted pre-processing code from [[3]](#3) and implemented the fine-tuning code for the RoBERTa model (to the time of writing, the code of the paper is not yet published).

Furthermore we used a published pre-trained RoBERTa model on the ARQMath dataset [[2]](#2) for fine-tuning.

## Prerequisites

#### Install dependencies

* pandas
* numpy
* sklearn
* huggingface transformers
* pytorch


#### Get data collection from Google Drive 
First create a new folder for the data collection:
mkdir data_collection

Then download the data collection from Google Drive:
* go to https://drive.google.com/drive/folders/1ZPKIWDnhMGRaPNVLi1reQxZWTfH2R4u3 (ARQMath dataset)
* right click on "collection"-folder -> Add shortcut to Drive
* All locations -> My Drive -> Add

!mkdir /content/ALBERT-for-Math-AR/raw_data

### Preprocess data collection
execute:
```bash
python3 preprocess.py
```
It merges all single xml files and convert the xml math notation to latex notation. The output is a json file which can be further processed.

### Create fine-tuning dataset train/dev/ 
execute:
```bash
python3 create_finetuning_dataset.py
```

## Training

To fine-tune the [pre-trained RoBERTa model]( https://huggingface.co/AnReu/math_pretrained_roberta)  on the ARQMath dataset, run this command:

```bash
python3 finetune.py
```



## References
<a id="1">[1]</a> Reusch, Anja and Thiele, Maik and Lehner, Wolfgang. An albert-based similarity measure for mathematical answer retrieval. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1593–1597, 2021.

<a id="2">[2]</a> Mansouri, Behrooz and Agarwal, Anurag and Oard, Douglas W and Zanibbi, Richard. Advancing math-aware search: The arqmath-3 lab at clef 2022. In European Conference on Information Retrieval, pages 408–415, 2022.

<a id="3">[3]</a> Reusch, Anja and Thiele, Maik and Lehner, Wolfgang. Transformer-encoder and decoder models for questions on math. In Proceedings of the Working Notes of CLEF, pages 5–8, 2022.</a>

 
