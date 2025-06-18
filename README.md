# UDA-DDA: Unsupervised Domain Adaptation with Dynamic Distribution Alignment Network For Emotion Recognition Using EEG Signals

This repository is the official implementation of My Paper Title. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset
This project uses the SEED, SEED_IV and DEAP dataset, which is publicly available for EEG-based emotion recognition. You can access the dataset from the following link:
[SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html). 
[SEED_IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html). 
[DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/). 

## Running the Code

To train the model(s) in the paper, run this command:

```train
python main.py
```

## Results

Our model achieves the following performance on :

| Dataset         | Accuracy  | STD |
| ------------------ |---------------- | -------------- |
|  SEED  |     87.27       |     07.55         |
|  SEED_IV  |     74.01       |     11.34         |

