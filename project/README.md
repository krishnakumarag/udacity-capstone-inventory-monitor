# Inventory Monitoring at Distribution Centers

In today world all are becoming online purchase. Here logistics and delivery system play important role. Therefore, the big warehouses require reliable and automated system to sort and distribute packages based on multiple characteristics. In these warehouses, packages/objects are usually carried over in boxes, where each box can hold multiple items. The task for the automated delivery system would be to classify the package and determine where to deliver it. But before that the system needs to know how many packages are in each delivery box and that will be the topic of the proposed Capstone project.

I want to use the  [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) to measure the number of objects in a box because it contains a picture of the contents of the box and information about other objects, such as how many there are. The adopted solution will determine the number of objects based on this image.


## Project Set Up and Installation
It is developed in AWS cloud services like s3, Sagemaker, Lambda and IAM. You Need an AWS account and clon [this repo](https://github.com/krishnakumarag/udacity-capstone-inventory-monitor.git). We used only free plan instances

Recommended machines are:
- for notebook: ml.t2.medium
- for training: ml.m5.large or ml.m5.xlarge
you can also use different instance types based on your need and cost. 

For more details refer [official AWS tutorial](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html)

## Project files
Project consist of multiple files:
- [`sagemaker.ipynb`](sagemaker.ipynb) -- main project file. Entrypoint
- [`hpo.py`](hpo.py) -- python script for Hyperparameter optimization using Sagemaker
- [`train.py`](train.py) -- python script for tuning the network. Can be used from Sagemaker or as standalone application
- [`inference.py`](inference.py) -- python script for running model inference
- [`file_list.json`](file_list.json) -- queried for the database to download only part of the dataset

## Dataset
As mentioned above we are using Amazon Bin Image Dataset which contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. You can see the data [here](https://registry.opendata.aws/amazon-bin-imagery/](https://registry.opendata.aws/amazon-bin-imagery/)


### Data selection
To speed up training process only a portion of data was selected from the dataset. 
- 1228 images with 1 items in it.
- 2299 images with 2 items in it.
- 2666 images with 3 items in it.
- 2373 images with 4 items in it.
- 1875 images with 5 items in it.

In total 10441 images were used. List of specific files is provided in `file_list.json` file.

### Data overview
Sample bin images:

![sample images in dataset](sample_images.png "Sample images in dataset")

### Data preprocessing
Downloaded data had to be divided into train and validation subsets. For this project images were divided as follows:
- Train: 60%
- Test: 20%
- Valid: 40%

### Access
For sagemaker training we use data from `S3`. So After preprocessing data are uploaded to `S3`.

## Model Training
As a baseline model used resnet50 image classification network. ResNet-50 is a convolutional neural network that is 50 layers deep. In AWS cloud there is available a pretrained version of the network trained on more than a million images from the [ImageNet database]http://www.image-net.org). The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.

### Hyperparameters tuning
Before actual training I tried to identify the best hyperparameters for the training job. For this I created `hpo.py` script which executes just a single epoch on a part of training data and tests following hyperparameter ranges:
- Learning rate was tuned for range: `(0.001, 0.1)` - found optimal value is `0.0012853059164119611`.
- Batch size was tuned for values: ``{32, 64, 128, 256, 512}`` - found optimal value is `32`.

### Model training procedure
After identification of potentially the best hyperparameters I ran training procedure for this task. The code for the training is provided in `train.py` file. The file is prepared to be working from Sagemaker notebook (example usage in `sagemaker.ipynb`) or as a standalone script which can run on your personal machine or on low-cost spot instances. For the 10000 files the training completed in 5 epochs after 2h of operation.

### Model evaluation and debugging
During training process SageMaker Debugger was enabled and generated following plot:
![Cross Entropy](CrossEntropyLoss_output.png "Cross Entropy plot")

Nevertheless, using manual debug logging and AWS CloudWatch service we can plot more detailed loss function plot

![Loss function - Cross Entropy](cross_entropy_loss.png "Loss function - Cross Entropy plot")

And accuracy dependency:

![Accuracy](accuracy.png "Accuracy")

As we can see Accuracy of trained network stabilized around value 0.3 - it is not perfect, but let's deploy the model.

## Model deployment

After training model can be deployed and used from different AWS services. Deployment procedure is presented in notebook `sagemaker.ipynb`.

## Model Inference
Using deployed model we can run prediction based on the source images. Let's use sample image such as:

![Test image](validated_image.jpg "Test image")

In this image network correctly predicted number of objects to be 2, which matches the image label. And our model predicted the correct result.

Example code for model inference:
```
from PIL import Image
import io

with open("validated_image.jpg", "rb") as image:
    f = image.read()
    img_bytes = bytearray(f)
    Image.open(io.BytesIO(img_bytes))
	
response=predictor.predict(img_bytes, initial_args={"ContentType": "image/jpeg"})
