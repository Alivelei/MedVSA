# MedVSA: Medical Visual Spoken-Question Answering
Two-stage Model

## Install

1. Clone this repository and navigate to CLS_adaptation folder

   ```
   https://github.com/Alivelei/MedVSW
   cd LA_MedVSW
   ```

   

2. Install Package: Create conda environment

   ```
   conda create -n medvsa python=3.10 -y
   conda activate medvsa
   pip install --upgrade pip
   pip install -r requirements.txt
   ```



# Model Download
To use it, you need to manually download the BiomedNLP-BiomedBERT-base, PMC-CLIP, and whisper-base checkpoints to the save folder.


# Data Download

https://drive.google.com/drive/folders/1JQovHtM9D0EwpdVnlZcFuaDy5c37LuU5


# Data Construction 

Store the datasets in the /data/ref/ directory. After downloading the datasets, please organize the directory structure as follows.

```
├── data_interface.py
├── datasets.py
├── __init__.py
├── ref 
│   ├── OVQA_publish
│   ├── PathVQA
│   ├── rad 
│   └── Slake1.0
├── tree.txt
└── word_sequence.py
```





# Training

```
# By adjusting the parameters in train.py, you can train various models across different datasets.
python train.py
```



# Testing

```
# By adjusting the test_best_model_path in test.py, you can load trained models for testing.
python test.py
```

