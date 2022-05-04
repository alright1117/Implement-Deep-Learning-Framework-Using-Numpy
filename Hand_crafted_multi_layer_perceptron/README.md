# Perceptron for image classfication 

## Overview 

This is the second homework in the deep learning course.

### Contents:
- [Overview](#overview)
- [Prerequisites](#Prerequisites)
- [Folder structure](#Folderstructure)
- [Dataset and preprocessing](#Datasetandpreprocessing)
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
>```images_feature/```      &nbsp; - &nbsp; the folder with HOG features <br/>

>```get_image_feature.py```     &nbsp; - &nbsp; extract the HOG feature from the dataset <br/>
>```nn.py```            &nbsp; - &nbsp; peceptron implementation <br/>
>```train_model.py```      &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model.py```       &nbsp; - &nbsp; applying the trained model to testing accuracy <br/>
>```model_pretrained.npy```     &nbsp; - &nbsp; pretrained model weight <br/>```plot.py```     &nbsp; - &nbsp; pretrained model weight <br/>

[[back]](#contents)
<br/>

---
### Dataset and preprocessing

- Download the [mini-imagenet](https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr "Title") dataset and put it into `images/` folder.     
- Use the following command to extract image features, the feature array will put into `images_feature/` folder.

```bash
python get_image_feature.py
```

[[back]](#contents)
<br/>

---
### Training

#### Start training
To train the model, use the following command:

```bash
python train_model.py dataset_dir=images_feature/ lr=0.1 input_size=3780 hidden_size=1000 output_size=50 iters=6000
```

Optional parameters (and default values):

>```dataset_dir```: **```images_feature/```** &nbsp; - &nbsp; path to the folder with the feature<br/>
>```lr```:  0.1&nbsp; - &nbsp; gradient descent learning rate <br/>```input_size```:  3780&nbsp; - &nbsp;input image features dimmension <br/>```hidden_size```:  1000&nbsp; - &nbsp; multilayer perceptron hidden dimmension <br/>```output_size```:  50&nbsp; - output classes num <br/>```iters```:  6000&nbsp; - &nbsp; the number of training iterations <br/>

[[back]](#contents)
<br/>

---
### Testing
To evaluate the top k accuracy output, use the following command. The command will load the pretrained model, and show the test result.

```bash
python test_model.py
```

[[back]](#contents)
</br>

---

