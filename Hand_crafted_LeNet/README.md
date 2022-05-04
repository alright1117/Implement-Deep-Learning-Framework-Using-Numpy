# Hand-crafted LeNet

## Overview 

An implementation of LeNet by numpy

### Contents:
- [Overview](#overview)
- [Prerequisites](#Prerequisites)
- [Folder structure](#Folderstructure)
- [Training](#Training)
- [Test](#Testing)

---
### Prerequisites:

- Python 3 
- numpy 1.19.4
- skimage 0.17.2

[[back]](#contents)
<br/>

---

### Folder structure

>```images/```             &nbsp; - &nbsp; the folder with the dataset <br/>

>```dataset.py```     &nbsp; - &nbsp; load and preprocess image for mini-ImageNet dataset <br/>
>```nn.py```            &nbsp; - &nbsp;layer implementation <br/>```optimizer.py```            &nbsp; - &nbsp;optimizer implementation <br/>```model.py```            &nbsp; - &nbsp;LeNet and LeNet Plus implementation <br/>```train.py```      &nbsp; - &nbsp;training model <br/>
>```test.py```       &nbsp; - &nbsp; test inference <br/>
>```LeNet.npy```     &nbsp; - &nbsp;LeNet pretrained weight <br/>```LeNet_plus.npy```     &nbsp; - &nbsp; LeNet Plus pretrained weight <br/>```plot.py```     &nbsp; - &nbsp; plot training loss and accuracy <br/>

[[back]](#contents)
<br/>

---
### Training

#### Start training
To train the model, use the following command:

```bash
python train_model.py --model LeNet --lr 0.001 --iters 10000 --train_batch 128 --val_batch 128
```

Optional parameters (and default values):

>```model```: **```LeNet```** &nbsp; - &nbsp; chose the model for training<br/>
>```lr```:  0.001&nbsp; - &nbsp;learning rate <br/>```iters```:  10000&nbsp; - &nbsp;the number of iteration <br/>```train_batch```:  128&nbsp; - &nbsp;the number of batch size for training <br/>```val_batch```:  128&nbsp; - &nbsp;the number of batch size for validation <br/>

[[back]](#contents)
<br/>

---
### Testing
To evaluate the test accuracy, use the following command. The command will load the pretrained model, and show the test result.

```bash
python test_model.py --model LeNet
```

Optional parameters (and default values):

>```model```: **```LeNet```** &nbsp; - &nbsp; chose the model for testing<br/>

[[back]](#contents)
</br>

---

