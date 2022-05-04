# Perceptron for image classfication 

## Overview 

This is the first homework in the deep learning course. It need to use hand-crafted feature for the mini-imagenet, and implement linear classifier for image classification. In oder to fulfill this requirements, I will use gradient decent to optimize model, and won’t use any packages with automatic differentiation such as pytorch and tensorflow.

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
- xgboost 1.3.1
- sklearn  0.19.1 (For random forest)

[[back]](#contents)
<br/>

---

### Folder structure

>```images/```             &nbsp; - &nbsp; the folder with the dataset <br/>
>```images_feature/```      &nbsp; - &nbsp; the folder with HOG features <br/>
>```results/```            &nbsp; - &nbsp; visual results for the training and validation accuracy <br/>

>```tree.ipynb```     &nbsp; - &nbsp; compare method for top k accuracy, you can see the result in this file. <br/>```get_image_feature.py```     &nbsp; - &nbsp; extract the HOG feature from the dataset <br/>
>```model.py```            &nbsp; - &nbsp; peceptron implementation <br/>
>```train_model.py```      &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model.py```       &nbsp; - &nbsp; applying the trained model to testing accuracy <br/>
>```model_pretrained.npy```     &nbsp; - &nbsp; pretrained model weight <br/>
>```train.txt```    &nbsp; - &nbsp; Training file name and label <br/>
>```val.txt```     &nbsp; - &nbsp; validation file name and label <br/>```test.txt```     &nbsp; - &nbsp; testing file name and label <br/>

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
python train_model.py dataset_dir=images_feature/ lr=0.01 layer=[3780,50] iters=20000
```

Optional parameters (and default values):

>```dataset_dir```: **```images_feature/```** &nbsp; - &nbsp; path to the folder with the feature<br/>
>```lr```:  0.01&nbsp; - &nbsp; gradient descent learning rate <br/>```layer```:  [3780, 50]&nbsp; - &nbsp; perceptron input shape and outputshape <br/>```iters```:  20000&nbsp; - &nbsp; the number of training iterations <br/>

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

