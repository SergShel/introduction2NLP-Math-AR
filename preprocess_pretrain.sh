#!/bin/bash
mkdir /content/introduction2NLP-Math-AR/data_processing
cd /content/introduction2NLP-Math-AR/preprocessing_scripts
git clone https://github.com/AnReu/ARQMathCode.git
python3 /content/introduction2NLP-Math-AR/preprocessing_scripts/get_clean_json.py
python3 /content/introduction2NLP-Math-AR/preprocessing_scripts/get_pretraining_data_base.py
python3 /content/introduction2NLP-Math-AR/preprocessing_scripts/get_pretraining_data_separated.py