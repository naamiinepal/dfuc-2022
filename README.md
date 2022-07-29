# **dfuc-2022**

# **Diabetic Foot Ulcer Segmentation**

**This work is a part of submission made to DFUC 2022. [DFUC 2022](https://dfuc2022.grand-challenge.org/DFUC2022) is hosted by [MICCAI 2022](https://conferences.miccai.org/2022/en), the 25th International Conference on Medical Image Computing and Computer Assisted Intervention.**

## **Usage**

**Note: Anybody can freely use this work under its license terms. However, the authors shall not be held accountable for the reliability and accuracy of the results and the consequences they may lead to.**

### **Requirements**
+ `python3`(This work has been developed using python 3.8.10)

### **Setup**

For setup and basic usage get this repository code in your local device and follow the instructions given.

**All the commands given are for linux system. For other system, please search for the equivalent commands accordingly.**

### **Create a virtual environment**

```python
python -m venv .venv
```
Note: This command assumes that you have `python3` as your default python version in your device.

### **Activate the environment**

```python
source .venv/bin/activate
```

### **Install the required dependencies**

```python
pip install -r requirements.txt
```

### **Start training**

For training, update the `train_config` file and run the command for training. Parameters in the `train_config` file can be updated by refering the documentation of training config at `docs/train_config_docs.md`.

```python
python train.py
```

Training logs can be visualized using `tensorboard`. It can be run using following command:

```
tensorboard --logdir logs/tensorboard/ --port PORT
```

Replace `PORT` with port number on which `tensorboard` is to be run. Visualizations can be obtained in web browser.

### **Evaluation**

For evaluation, update the `eval_config` file and run the command for evaluation. Parameters in the `eval_config` file can be updated by refering the documentation of evaluation config at `docs/eval_config_docs.md`.

```python
python eval.py
```

## **License**
This work is available under `MIT License`. See `LICENSE` for the full license text.
