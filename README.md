[![en](https://img.shields.io/badge/lang-en-green.svg)](README.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](README.ru.md)
# Skillbox. Thesis on computer vision. Human Emotion Recognition

As part of this thesis, a notebook was developed that can be used to create a ready-to-use model for recognizing human emotions based on facial expressions. 
The work was carried out as part of the [skillbox-computer-vision-project](https://www.kaggle.com/c/skillbox-computer-vision-project) competition on the [Kaggle](https://www.kaggle.com) platform.

The notebook allows to create an emotion recognition model of one of the following types:
#### 1. A model that predicts the probabilities of emotions that a person is experiencing based on their facial expression.
#### 2. A model that can recognize the valence and intensity levels of an emotion that a person is experiencing (on a scale from -1 to +1) based on their facial expression.

The type of the created model is determined by the way the emotions that the model should recognize are described in the project settings. If emotions are described by a simple list or a tuple of their names (recognized classes), then a of the [1st type](#1-a-model-that-predicts-the-probabilities-of-emotions-that-a-person-is-experiencing-based-on-their-facial-expression). ​​If the description of emotions is presented by a dictionary in which each emotion is matched with a pair of values ​​characterizing typical levels of valence and intensity of this emotion, then a model of the [2nd type](#2-a-model-that-can-recognize-the-valence-and-intensity-levels-of-an-emotion-that-a-person-is-experiencing-on-a-scale-from--1-to-1-based-on-their-facial-expression).

<details><summary>Examples of emotion descriptions</summary>
<p>
<table>
<tr>
<td>
<p align="center"><b>1st type model</b></p>

```python
EMOTIONS = (
    'anger',
    'contempt',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise',
    'uncertain',
)
```

</td>
<td>
<p align="center"><b>2nd type model</b></p>

```python
EMOTIONS = {
    'anger': (-0.41, 0.79),
    'contempt': (-0.57, 0.66),
    'disgust': (-0.67, 0.49),
    'fear': (-0.12, 0.78),
    'happy': (0.9, 0.16),
    'neutral': (0.0, 0.0),
    'sad': (-0.82, -0.4),
    'surprise': (0.37, 0.91),
    'uncertain': (-0.5, 0.0),
}
```

</td>
</tr>
</table>
</p>
</details>

The model is created using the [Transfer Learning](https://keras.io/guides/transfer_learning/) and [Fine Tuning](https://keras.io/guides/transfer_learning/) mechanisms. That is, the model is not created from scratch, but based on a trained image classification model, the so-called base model. Any model from the [Keras Applications](https://keras.io/api/applications/) library trained on the [ImageNet](https://www.image-net.org/) dataset can serve as a base model.

The base model is selected during the notebook execution based on the accuracy of its predictions on the [ImageNet](https://www.image-net.org/) validation dataset, its size, and the measured inference speed of the resulting model built on the basis of this base model (for more details, see the description of the ["Base Model Selection"](#23-%D0%B2%D1%8B%D0%B1%D0%BE%D1%80-%D0%B1%D0%B0%D0%B7%D0%BE%D0%B2%D0%BE%D0%B9-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8-base_model_selection) stage). The list of base models from which the selection is made is specified in the project settings as a dictionary, in which the name of each base model is assigned a pair of reference values: its size in MB and its accuracy on the [ImageNet](https://www.image-net.org/) validation dataset in percent (see the ["Available models"](https://keras.io/api/applications/) table, the "Size (MB)" and "Top-1 Accuracy" columns, respectively).

<details><summary>Example list of base models</summary>
<p>
    
```python
KERAS_BASE_MODELS = {
    'MobileNet': (16, 70.40),
    'MobileNetV2': (14, 71.30),
    'NASNetMobile': (23, 74.40),
    'InceptionV3': (92, 77.90),
    'ResNet50V2': (98, 76.00),
    'EfficientNetB0': (29, 77.10),
    'ResNet50': (98, 74.90),
    'EfficientNetB1': (31, 79.10),
    'VGG16': (528, 71.30),
    'ResNet101V2': (171, 77.20),
    'DenseNet121': (33, 75.00),
    'EfficientNetB2': (36, 80.10),
    'VGG19': (549, 71.30),
    'ResNet101': (171, 76.40),
    'DenseNet169': (57, 76.20),
    'ResNet152V2': (232, 78.00),
    'Xception': (88, 79.00),
    'DenseNet201': (80, 77.30),
    'ResNet152': (232, 76.60),
    'InceptionResNetV2': (215, 80.30),
    'EfficientNetB3': (48, 81.60),
    'EfficientNetB4': (75, 82.90),
    'NASNetLarge': (343, 82.50),
    'EfficientNetB5': (118, 83.60),
    'EfficientNetB6': (166, 84.00),
    'EfficientNetB7': (256, 84.30),
    'EfficientNetV2B0': (29, 78.70),
    'EfficientNetV2B1': (34, 79.80),
    'EfficientNetV2B2': (42, 80.50),
    'EfficientNetV2B3': (59, 82.00),
    'EfficientNetV2S': (88, 83.90),
    'EfficientNetV2M': (220, 85.30),
    'EfficientNetV2L': (479, 85.70),
}
```
    
</p>
</details>

## Model architecture

The base model is trained to determine which object from the [ImageNet](https://www.image-net.org/) dataset is presented in the input image based on its features. The convolutional part of the model is responsible for feature extraction. The last, fully connected layer (*dense layer*) with the [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) activation function, located as if on top of the model, is responsible for identifying the object based on its features. Since the model being built solves a completely different problem, this fully connected layer is not needed in the resulting model. Therefore, only the convolutional part of the base model is used. At the top of the convolutional part is the pooling layer. The default pooling type is average pooling, but maximum pooling can also be used. The type of pooling used is determined by the user and is specified in the project settings.

<details><summary>Example of a pooling setup</summary>
<p>

```python
BASE_MODEL_POOLINGS = 'avg' # Type of pooling at the output of the base models ('avg' - average, 'max' - max)
```

</p>
</details>

To identify emotions based on the features obtained using the base model, one or more fully connected layers with a [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) type activation function can be added on top of it. In addition, to ensure regularization during the training process, a dropout layer is placed before each such layer. During the training process, such a layer randomly zeroes a given portion of the input neurons. At the same time, the remaining input neurons are proportionally increased so that the sum of all input neurons does not change.

The user can specify different options for the number of such layer blocks (dropout layer + fully connected layer), options for the number of output neurons in fully connected layers, as well as options for the values ​​of the proportion of neurons to be zeroed in dropout layers. During the execution of the notebook, the optimal one will be found from these variants (for more details, see step ["3.3. Selecting the best fully connected model"](#33-%D0%B2%D1%8B%D0%B1%D0%BE%D1%80-%D0%BB%D1%83%D1%87%D1%88%D0%B5%D0%B9-%D0%BF%D0%BE%D0%BB%D0%BD%D0%BE%D1%81%D0%B2%D1%8F%D0%B7%D0%BD%D0%BE%D0%B9-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8-model_on_top_selection)).

All of the listed layers form the conventionally named "on top model".

<details><summary>Example of setting configuration options for blocks from dropout layers and fully connected layers</summary>
<p>
    
```python
MODEL_ON_TOP_DENSE_NUMS = [1, 2] # Options for the number of additional fully connected layers
MODEL_ON_TOP_DENSE_UNITS = [1024, 2048] # Options for the number of output neurons in the additional fully connected layer
MODEL_ON_TOP_DROPOUT_RATES = [.0, .2] # Options for the proportion of data to drop before feeding into the fully connected layer during training
```
    
</p>
</details>

When constructing the [1st type] (#1-a-model-that-predicts-the-probabilities-of-emotions-that-a-person-is-experiencing-based-on-their-facial-expression) model another fully connected layer with the function is added to the last block activation of the [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) model and the number of output neurons corresponding to the number of recognized emotions. In this case, at the output of the model, during inference, a vector of probabilities is formed for what emotion the person's face expresses at the input image.

When constructing the model [2nd type](#2-a-model-that-can-recognize-the-valence-and-intensity-levels-of-an-emotion-that-a-person-is-experiencing-on-a-scale-from--1-to-1-based-on-their-facial-expression) model another fully connected layer is also added to the last block. But this layer has only 2 output neurons, and the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function is used for activation, with a value limit of up to 2. Thus, the output neurons of this layer can take values ​​in the range from 0 to 2. But since the predicted values ​​of valence and intensity must be in the range from -1 to 1, then at the end of the model another layer is placed, which subtracts 1 from the values ​​of the output neurons of the previous fully connected layer.

Additionally in In order to improve the quality of model training by increasing the diversity of input images, an augmentation model is added to the base model. This model randomly transforms the input image during training before feeding it to the base model. The augmentation model randomly rotates the image slightly , changes its contrast and brightness, and also mirrors it. The user can define the ranges of these transformations in the project settings.

<details><summary>Пример настройки аугментации входных изображений</summary>
<p>
    
```python
RANDOM_FLIP = 'horizontal' # Random image flip type
RANDOM_ZOOM = .2 # Maximum image scale change
RANDOM_ROTATION_FACTOR = .1 # Maximum image rotation (in fractions of a full rotation - 360°)
RANDOM_CONTRACT_FACTOR = .2 # Maximum contrast change (in fractions of the original value)
RANDOM_BRIGHTNESS_FACTOR = .2 # Maximum brightness change (in fractions of the original value)
```
    
</p>
</details>

Thus, the resulting model during training consists of three sequentially connected models:
- augmentation model;
- base model;
- on top model.

<details><summary>Structural diagrams of the model in training mode</summary>
<p>
<tr>
<td>
<p align="center"><b>1st type model</b></p>
<img width=100% src=https://user-images.githubusercontent.com/107345313/200283993-4ec70b6c-9da7-4355-891a-564332559041.svg>
<p></p>
</td>
</tr>
<tr>
<td>
<p align="center"><b>2nd type model</b></p>
<img width=100% src=https://user-images.githubusercontent.com/107345313/200284152-5faabe63-43ab-4593-828b-2684ef35b70f.svg>
<p></p>
</td>
</tr>
</p>
</details>

As noted above, the augmentation model and the exclusion layers in the upper model do not participate in the operation of the model in the inference mode. That is, the input image is fed directly to the input of the base model and the input neurons of the fully connected layers of the upper model are not zeroed. Therefore, the structure of the model in the inference mode can be presented in a simplified form without these elements.

<details><summary>Structural diagrams of the model in inference mode</summary>
<p>
<tr>
<td>
<p align="center"><b>1st type model</b></p>
<img  width=100% src=https://user-images.githubusercontent.com/107345313/200292114-4c6a4e79-a151-463c-a3d9-e22bfa526efa.svg>
<p></p>
</td>
</tr>
<tr>
<td>    
<p align="center"><b>2nd type model</b></p>
<img  width=100% src=https://user-images.githubusercontent.com/107345313/200292181-72ec54e2-f367-4c06-854f-ce541722d108.svg>
<p></p>
</td>
</tr>
</p>
</details>

## Class for implementing a trained model

A special class `FaceEmotionRecognitionNet` is provided for using a trained model as part of an emotion recognition application. This class prepares the input face image before feeding it to the model, obtains model predictions, and extracts meaningful information from the model predictions.

When creating a class, you need to pass the path to the trained model file (the `file_path` parameter) and a description of the emotions recognized by the model (`emotions`). The type of emotion description shows the class what type of model is used. If emotions are described using a simple list or a tuple of their names, then this is a model of the [1st type](#1-a-model-that-predicts-the-probabilities-of-emotions-that-a-person-is-experiencing-based-on-their-facial-expression). ​​If the description of emotions is presented by a dictionary in which each emotion is matched with a pair of values ​​characterizing the typical levels of valence and intensity of this emotion, then this is a model of the [2nd type](#2-a-model-that-can-recognize-the-valence-and-intensity-levels-of-an-emotion-that-a-person-is-experiencing-on-a-scale-from--1-to-1-based-on-their-facial-expression).

This class has only one method - `predict`, which, based on an array representing a face image (argument `face_image`), depending on the model type, returns:
- [1st type](#1-a-model-that-predicts-the-probabilities-of-emotions-that-a-person-is-experiencing-based-on-their-facial-expression) - the name of the emotion (`emotion`) that the face in the input image most likely expresses, and the probability (`probability`) with which the model recognized this emotion;
- [2nd type](#2-a-model-that-can-recognize-the-valence-and-intensity-levels-of-an-emotion-that-a-person-is-experiencing-on-a-scale-from--1-to-1-based-on-their-facial-expression)- the name of the emotion (`emotion`) whose typical valence and intensity values ​​are closest to the values ​​predicted by the model, the distance between them (`error`) and the valence and intensity (`arousal`) values ​​predicted by the model of the emotion expressed by the face in the input image.
- 
## Data for creating the model

### Training data

The dataset attached to the [skillbox-computer-vision-project](https://www.kaggle.com/competitions/skillbox-computer-vision-project/data) competition is used to train the model. The dataset is a set of files distributed across folders with emotion names. However, any other emotion dataset with the same markup can be used for training. Such a dataset should be placed as an archive in the [Google Drive](https://drive.google.com/) cloud storage. When running the notebook, this dataset is downloaded from the link specified in the project settings and extracted from the archive to the project folder.

<details><summary>Example of specifying a link to a training dataset</summary>
<p>
    
```python
TRAIN_DATASET_URL = 'https://drive.google.com/file/d/1TG9P5B2k3eTbC4XDxDmEc07dyAORPC16/view?usp=sharing' # Link to a training dataset
TRAIN_DATASET_EXT = 'zip' # Type (file extension) of the training dataset archive
```
    
</p>
</details>

### Test data

To evaluate the quality of the model during the training process, the dataset attached to the competition [skillbox-computer-vision-project](https://www.kaggle.com/competitions/skillbox-computer-vision-project/data) is also used. The dataset is also a set of files. But this dataset has no markup, and the quality of the model can only be assessed on the platform itself. When running a notebook, this dataset is also downloaded from the link specified in the project settings and extracted from the archive to the project folder.

<details><summary>Example of specifying a link to a test dataset</summary>
<p>
    
```python
TEST_DATASET_URL = 'https://drive.google.com/file/d/12QrDrLT1F-X7UycvOoApXFqxTw3Zx93K/view?usp=sharing' # Link to a test dataset
TEST_DATASET_EXT = 'zip' # Type (file extension) of the test dataset archive
```
    
</p>
</details>

### Model Quality Assessment

As stated above, the [Kaggle](https://www.kaggle.com/) platform is used to check the accuracy of predictions on the test dataset. For this purpose, a special `Kaggle` class was created in the notebook, which, using the [Kaggle API](https://github.com/Kaggle/kaggle-api), implements sending a file with the model predictions for verification in the form of a csv file and receiving the results of checking the public (Public) and private (Private) parts of the predictions. The average value of these assessments is taken as the metric for the quality of the model. To connect the notebook to the platform, an `API Token` is required as a json file. The user must first copy it from the platform in the profile editing section. The resulting `API Token` file must be placed in the [Google Drive](https://drive.google.com/) cloud storage and a link to it must be specified in the project settings.

<details><summary>Example of specifying a link to a token file [Kaggle API](https://github.com/Kaggle/kaggle-api)</summary>
<p>
    
```python
KAGGLE_API_TOKEN_URL = 'https://drive.google.com/file/d/*********************************/view?usp=sharing' # Link to the token for connecting to the Kaggle platform via API
```

</p>
</details>

In addition, the user must register as a participant in the [skillbox-computer-vision-project](https://www.kaggle.com/c/skillbox-computer-vision-project) competition.

## Pipelines
The model creation process is divided into three independent stages (pipelines), which are executed sequentially:
1. Collecting information about base models in [Keras Applications](https://keras.io/api/applications/) (`KERAS_BASE_MODELS_PROCESSING_PIPELINE`).
2. Image preprocessing (`IMAGE_PREPROCESSING_PIPELINE`).
3. Model creation (`MODEL_BUILDING_PIPELINE`).

The results of the pipeline stages are saved in csv files in the pipeline folder in the shared storage. The names of the pipeline folders match the names of the pipelines specified in the pipeline settings (the `name` parameter). The names of the report csv files are also specified in the pipeline settings (the `report_csv` parameter).

<details><summary>Pipeline execution log file fields</summary>
<p>

- `stage` - stage name;
- `params` - stage configuration parameters;
- `platform` - stage execution platform;
- `start_time` - stage execution start time;
- `update_time` - stage status update time;
- `state` - stage execution status:
    * `skipped (not ready)` - skipped due to not being ready (previous stage(s) not executed);
    * `skipped (platform)` - skipped due to platform mismatch (pipeline must be executed on a different platform);
    * `run started` - stage iteration is in progress;
    * `started` - stage is in progress;
    * `complete` - stage is complete.

</p>
</details>

Each pipeline in turn consists of several stages, which are also executed sequentially. The pipeline stage settings are specified in the pipeline settings as a list (the `stages` parameter), which also determines the sequence of their execution.

The settings for each stage include its name (the `name` parameter) and a specific set of its configuration parameters (the `params` parameter). In addition, any stage, at the user's discretion, can be executed either on a local computer in [JupyterNotebook](https://jupyter.org/) or [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html), or remotely in [Google Colab](https://colab.research.google.com/). The platform for executing the stage is also specified in its settings (the `platform` parameter). This option allows you to execute stages on the most suitable platform for this. So, the stages during which inference or training of the basic part of the model is performed are best performed on [Google Colab](https://colab.research.google.com/) using a hardware accelerator GPU or TPU. And the remaining stages may be faster on a local computer.

In order to facilitate switching between platforms, all data is saved in a single cloud storage [Google Drive](https://drive.google.com/). So, all the data required to perform a pipeline stage is downloaded to the project folder on the execution platform from a single storage. And upon completion of the stage, the protocol and the results of its execution are saved as csv files in the pipeline folder. The resulting datasets, model training logs and the models themselves are saved in the project folder in a single storage as zip archives.

Before you start working with your notebook, you need to install the [Google Drive](https://www.google.com/intl/ru_ru/drive/download/) application on your local computer and connect the user's cloud storage as a logical drive on your local computer.

<details><summary>Example of project folder path settings</summary>
<p>

```python
PROJECT_NAME = 'skillbox-computer-vision-project' # Project name
LOCAL_PROJ_PATH = f'D:/{PROJECT_NAME}' # Path to the project folder on the local computer
COLAB_PROJ_PATH = f'/content/{PROJECT_NAME}' # Path to the project folder in the Google Colab session storage
LOCAL_GD_PROJ_PATH = f'G:/My Drive/{PROJECT_NAME}' # Path to the project folder on Google Drive on the local computer
COLAB_GD_PROJ_PATH = f'/content/drive/MyDrive/{PROJECT_NAME}' # Path to the project folder on Google Drive in Google Colab
```

</p>
</details>

In addition, to improve the convenience of using the notebook, each time it is started, the packages necessary for its operation on a specific platform are automatically installed.

### 1. Collecting information about base models in [Keras Applications](https://keras.io/api/applications/) (`KERAS_BASE_MODELS_PROCESSING_PIPELINE`)

<details><summary>Pipeline setup example</summary>
<p>

```python
KERAS_BASE_MODELS_PROCESSING_PIPELINE = {
    'name': 'keras_base_models_processing',
    'description': 'Pipeline for collecting information about base models in Keras Applications',
    'report_csv': 'pipeline_base_models_processing.csv',
    'stages': [
        {
            'name': 'sizes_retrieving',
            'description': 'Getting information about the sizes of input images and feature vectors',
            'platform': 'colab', # Runs in Google Colab
            'params': {
                'result_csv': 'base_model_sizes.csv', # Path to the file with the selected models
            }
        },
        {
            'name': 'inference_time_measuring',
            'description': 'Measuring model inference time',
            'platform': 'colab', # Runs in Google Colab
            'params': {
                'batch_size': 1, # Batch size
                'batches': 1, # Number of batches in the dataset
                'repetitions': 100, # Number of repetitions
                'result_csv': 'model_inference_times.csv', # Path to the file with the selected models
            }
        },
        {
        'name': 'base_model_selection',
        'description': 'Base model selection',
        'platform': 'colab', # Runs in Google Colab
        'params': {
                'inference_time_weight': INFERENCE_TIME_WEIGHT, # Inference time weight when selecting a base model
                'top1_accuracy_weight': 1 - INFERENCE_TIME_WEIGHT, # Accuracy weight when selecting a base model
                'process_csv': 'base_model_selection.csv', # Path to the file with the base model selection process data
                'result_csv': 'base_model.csv', # Path to the file with the description of the selected base model
            }
        },
    ]
}
```
    
</p>
</details>

#### 1.1. Retrieving information about the sizes of input images and feature vectors (`sizes_retrieving`)
The first goal of this stage is to obtain information about the optimal sizes of input images for each base model. Images fed to the model during training and use for obtaining predictions will be scaled to this size.

The second goal is to obtain information about the sizes of feature vectors at the output of the last pooling layer of each base model. This information is needed to create the upper model, which provides the transformation of the feature vector, depending on the model type, either into a vector of emotion probabilities or into a pair of valence and intensity values.

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`keras_base_models_processing/base_model_sizes.csv`:
- `base_model_name` - the name of the base model;
- `image_size` - the optimal size of the input image;
- `feature_size` - feature vector size.

</p>
</details>

#### 1.2. Measuring inference time (`inference_time_measuring`)
The goal of this step is to estimate the inference time of a model consisting of a base model combined with the "heaviest" upper model (the model containing the maximum number of fully connected layers of the maximum size). This information allows us to exclude base models with insufficient performance from further consideration.

<details><summary>Fields of the csv file of the step execution results</summary>
<p>

`keras_base_models_processing/model_inference_times.csv`:
- `base_model_name` - the name of the base model;
- `inference_time` - the average inference time of the model in seconds.

</p>
</details>

#### 1.3. Base model selection (`base_model_selection`)
The goal of this stage is to select a base model that will be used to build the resulting model. The selection is based on the best combination of performance and accuracy of the model shown on the [ImageNet](https://www.image-net.org/) validation dataset, taking into account the weights of these metrics specified by the user (parameters `inference_time_weight` and `top1_accuracy_weight`, respectively).

In addition to the above metrics, the selection of the base model is affected by its size. The user can limit the range of models considered by setting the maximum allowable model size in MB (parameter `BASE_MODEL_MAX_SIZE` in the "Basic settings" section).

For each selected model, points are awarded for accuracy and points for performance. Accuracy points are awarded according to the following rule:

$$Sa_i = {a_i - min(a_1, ..., a_n)\over max(a_1, ..., a_n) - min(a_1, ..., a_n)},$$
where&nbsp; $a_i$ is the accuracy of the model being evaluated;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$a_1, ..., a_n$ is the accuracy of the models being evaluated.

Performance points are awarded according to the following rule:
$$St_i = {max(t_1, ..., t_n) - t_i \over max(t_1, ..., t_n) - min(t_1, ..., t_n)}$$
where&nbsp; $t_i$ is the inference time of the model being evaluated;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $t_1, ..., t_n$ - inference time of the models being evaluated.

The total weighted points for accuracy and performance take into account the ratio of the importance of accuracy and performance of the model and are determined by the following rule.
$$S_i = Sa_i Wa + St_i Wt $$
where&nbsp; $Sa_i$ - points for the accuracy of the model being evaluated;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $St_i$ - points for the performance of the model being evaluated;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $Wa$ - weight of the accuracy score;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $Wt$ - the weight of the performance score.

For further work, the model with rank #1 is selected, i.e. the best weighted score for accuracy and performance.

Its name and main characteristics are saved in the resulting csv file.

<details><summary>Fields of the stage execution report csv file</summary>
<p>

`keras_base_models_processing/base_model_selection.csv`:
- `base_model_name` - base model name,
- `top1_accuracy` - base model accuracy on the [ImageNet](https://www.image-net.org/) validation dataset,
- `size` - base model size in MB,
- `colab_inference_time` - inference time measured in the previous stage in seconds,
- `top1_accuracy_score` - points for accuracy,
- `colab_inference_time_score` - points for performance,
- `weighted_score` - weighted points for accuracy and performance,
- `rank` - model compliance rank.

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`keras_base_models_processing/base_model.csv`:
- `base_model_name` - the name of the selected base model;
- `image_size` - the size of the input image of the selected base model;
- `feature_size` - the size of the feature vector of the selected base model.

</p>
</details>

### 2. Image preprocessing (`IMAGE_PREPROCESSING_PIPELINE`).
This pipeline is designed to prepare training and test datasets. During the pipeline execution, images are cropped to the face area to remove unnecessary details that may "confuse" the model. Additionally, images of faces that are too similar to each other and images of faces whose labeling is questionable are excluded from the training dataset.

Faces in images are searched using the pretrained face detector [MTCNN](https://github.com/ipazc/mtcnn) by Iván de Paz Centeno <ipazc@unileon.es>. The detector is built on the basis of a multi-task convolutional neural network (*Multi-task Cascaded Convolutional Network*), described in the article by Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li and Yu Qiao "[Joint face detection and alignment using multitask cascaded convolutional networks](https://arxiv.org/abs/1604.02878)".

<details><summary>Example of setting up a pipeline</summary>
<p>

```python
IMAGE_PREPROCESSING_PIPELINE = {
    'name': 'image_preprocessing',
    'description': 'Image preprocessing pipeline',
    'report_csv': 'pipeline_images_preprocessing.csv',
    'stages':
    [
        {
            'name': 'train_face_extraction',
            'description': 'Extracting face images from the training dataset',
            'platform': 'local', # Runs on the local computer
            'params': {
                'path': 'train_faces', # Path to the folder of the training dataset with face images
                'engines': 8, # Number of parallel running "engines"
                'batch_size': 125, # Batch size
                'scale_factor': 0.709, # Scaling factor for face detection in an image
                'min_face_size': 128, # Minimum face size for face detection in an image
                'process_csv': 'train_face_extraction_process.csv', # Path to the file with detailed information
                'result_csv': 'train_face_extraction.csv', # Path to the file with results
            },
        },
        {
            'name': 'test_face_extraction',
            'description': 'Extracting face images from the test dataset',
            'platform': 'local', # Executed on the local computer
            'params': {
                'path': 'test_faces', # Path to the test dataset folder dataset with face images
                'engines': 8, # Number of parallel running "engines"
                'batch_size': 125, # Batch size
                'scale_factor': 0.709, # Scaling factor for face detection in an image
                'min_face_size': 128, # Minimum face size for face detection in an image
                'process_csv': 'test_face_extraction_process.csv', # Path to file with detailed information
                'result_csv': 'test_face_extraction.csv', # Path to file with results
            }
        },
        {
            'name': 'train_cleaning',
            'description': 'Additional cleaning of training dataset',
            'platform': 'colab', # Performed in Google Colab
            'params': {
                'path': 'train_clean_faces', # Path to the folder of the cleaned training dataset
                'batch_size': 64, # Batch size
                'buffer_size': 10, # Buffer size
                'process_csv': 'train_cleaning_process.csv', # Path to the file with detailed information
                'result_csv': 'train_cleaning.csv', # Name of the file with results
            }
        },
    ]
}
```

</p>
</details>

#### 2.1. Extracting face images from the training dataset (`train_face_extraction`)

This stage has two goals:
1. Remove unnecessary details from the images that can "confuse" the model, leaving only the face areas.
2. Exclude from the dataset images that do not contain faces or that show them partially.

As a result of this stage, a dataset of extracted face images is created, the zip archive of which is saved in a single cloud storage. The path to the dataset in the project folder on the execution platform and, accordingly, the name of the dataset archive in the single cloud storage are determined by the `path` parameter in the stage settings.

In order to optimally use the platform's computing resources and reduce the pipeline execution time, all stages are performed using a parallel computing mechanism implemented using the [IPython Parallel](https://ipyparallel.readthedocs.io/en/latest/) package. The number of parallel "engines" (engines) is specified by the `engines` parameter and should be selected based on the available computing resources of the execution platform.

After completing the stage, it is necessary to visually check the quality of the resulting dataset of face images. If a large number of images are excluded or, conversely, if the dataset contains images that are not human faces, it is necessary to adjust the detector settings (parameters `scale_factor` and `min_face_size`) and run it again.

<details><summary>Fields of the CSV file of the stage execution report</summary>
<p>

`image_preprocessing/train_face_extraction_process.csv`:
- `file_path` - relative path to the source dataset file;
- `image_size` - image size (maximum of width and height);
- `faces_num` - number of detected images;
- `face_size` - the size of the first found face image (maximum of the width and height);
- `batch` - the batch sequence number that includes the original image;
- `iter` - the batch processing iteration number
- `engine` - the sequence number of the cluster "engine" in which the batch is processed
- `status` - the processing execution status: ok-successful; error-error

</p>
</details>

<details><summary>Fields of the CSV file of the stage execution results</summary>
<p>

`image_preprocessing/train_face_extraction.csv`:
- `emotion` - the name of the emotion;
- `failed_images_number` - the number of files in the original dataset where face images could not be found;
- `faces_num` - the number of face images with an expression of a certain emotion.

</p>
</details>

#### 2.2. Extracting face images from the test dataset (`test_face_extraction`)
The purpose of this stage is only to extract faces from the test dataset (cutting off unnecessary information). It is assumed that the test data is a priori correct, i.e. all images contain a complete image of at least one face. Therefore, if a face is not detected on at least one source image, the stage is considered not completed and it is necessary to adjust the detector settings and run it again.

Similar to [previous](#21-extracting-face-images-from-the-training-dataset-train_face_extraction) stage during the execution of the stage, an archive of the test dataset of face images and csv files of the report and results of the stage are formed.

<details><summary>Fields of the CSV file of the stage execution report</summary>
<p>

`image_preprocessing/test_face_extraction_process.csv`:
- `file_path` - relative path to the source dataset file;
- `image_size` - image size (maximum of width and height);
- `faces_num` - number of detected images;
- `face_size` - image size of the first found face (maximum of width and height);
- `batch` - batch number of the batch that contains the source image;
- `iter` - batch processing iteration number
- `engine` - sequence number of the cluster "engine" in which the batch is processed
- `status` - processing execution status: ok-

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`image_preprocessing/test_face_extraction.csv`:
- `emotion` - emotion name;
- `failed_images_number` - number of files of the original dataset where a face image could not be found;
- `faces_num` - number of face images with a certain emotion.

</p>
</details>

#### 2.3. Cleaning the training dataset (`train_cleaning`)
The goal of this stage is to exclude from the training dataset of face images objects that may negatively affect the quality of model training:
- identical or very similar face images;
- face images that may have been incorrectly labeled.

The above-mentioned face images are identified among images with the same labeling (within a class) based on the analysis of their feature vectors obtained using the base model. Feature vectors are obtained by "passing" images through the base model. Next, for each pair of obtained feature vectors of face images with the same labeling, the [cosine distance](https://wiki5.ru/wiki/Cosine_similarity) (similarity measure) between them is calculated.

Too similar face images are identified according to the following rule. Such images include pairs of images, the [cosine distance](https://wiki5.ru/wiki/Cosine_similarity) between which exceeds the upper reliability limit. This limit is 1.5 interquartile ranges above the 3rd quartile:

$$\cos(\theta)_{max}=Q_3+1.5\cdot(Q_3-Q_1)$$,
where&nbsp; $Q_1$ is the 1st quartile of the distribution of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between pairs of feature vectors of face images with the same
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; markup;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $Q_3$ is the 3rd quartile of the distribution of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between pairs of feature vectors of face images with the same
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;markup.

To identify too different face images, the median value of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between the feature vector of this image and the feature vectors of the other face images with the same markup is additionally calculated for each image. Images that are too dissimilar are those whose median similarity value with other images falls outside the lower confidence limit, which is 1.5 interquartile ranges below the 1st quartile:

$$Me_{min}=Q_1-1.5\cdot(Q_3-Q_1),$$
where&nbsp; $Q_1$ is the 1st quartile of the distribution of median values ​​[cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between the feature vector of the face image and the feature vectors of the other faces with the same markup;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $Q_3$ - 3rd quartile of the distribution of median values ​​of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between the face image feature vector and &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;feature vectors of other faces with the same markup.

<details><summary>Fields of the CSV file of the stage completion report</summary>
<p>

`image_preprocessing/train_cleaning_process.csv`:
- `file_path` - relative path to the file in the face image dataset;
- `similarity_max` - upper limit of the statistically significant sample of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between the image feature vectors;
- `similarity` - a list of values ​​of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) of the feature vector relative to other vectors that are beyond the upper limit of the statistically significant sample;
- `similar_to` - a list of relative paths to the face image files with which excessive similarity was detected;
- `similarity_median_min` - the lower limit of the statistically significant sample of median values ​​of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between the feature vector of images and the feature vectors of the remaining images;
- `similarity_median` - the median value of [cosine distances](https://wiki5.ru/wiki/Cosine_similarity) between the feature vector of an image and the feature vectors of the remaining faces expressing the same emotion.

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`image_preprocessing/train_cleaning.csv`:
- `emotion` - the name of the emotion;
- `failed_faces_number` - the number of rejected face images;
- `faces_num` - the remaining number of face images.

</p>
</details>

### 3. Creating a model (`MODEL_BUILDING_PIPELINE`)

<details><summary>Example of setting up a pipeline</summary>
<p>

```python
MODEL_BUILDING_PIPELINE = {
'name': 'model_building',
'description': 'Model building pipeline',
'report_csv': 'pipeline_model_building.csv',
'stages': [
        {
            'name': 'train_feature_extraction',
            'description': 'Extracting features from the training dataset',
            'platform': 'colab', # Performed in Google Colab
            'params': {
                'path': 'train_features', # Path to the folder with the extracted feature batch files
                'flip': RANDOM_FLIP, # Random image cropping
                'rotation_factor': RANDOM_ROTATION_FACTOR, # Factor of random rotation (counter-clockwise or clockwise) of the image when augmentations, fractions of 360°
                'zoom_factor': RANDOM_ZOOM, # Factor of random zooming in or out of an image during augmentation
                'contrast_factor': RANDOM_CONTRACT_FACTOR, # Factor of random change of image contrast
                'brightness_factor': RANDOM_BRIGHTNESS_FACTOR, # Factor of random change of image brightness
                'batch_size': 64, # Batch size
                'buffer_size': 10, # Buffer size
            }
        },
        {
            'name': 'test_feature_extraction',
            'description': 'Extracting features from a test dataset',
            'platform': 'colab', # Performed in Google Colab
            'params': {
                'path': 'test_features', # Path to the folder with the extracted feature batch files
                'flip': RANDOM_FLIP, # Random image cropping
                'rotation_factor': RANDOM_ROTATION_FACTOR, # Factor of random rotation (counter-clockwise or clockwise) of the image during augmentation, fraction of 360°
                'zoom_factor': RANDOM_ZOOM, # Factor of random zooming in or out of the image during augmentation
                'contrast_factor': RANDOM_CONTRACT_FACTOR, # Factor of random change in image contrast
                'brightness_factor': RANDOM_BRIGHTNESS_FACTOR, # Factor of random change of image brightness
                'batch_size': 64, # Batch size
                'buffer_size': 10, # Buffer size
            }
        },
        {
            'name': 'model_on_top_selection',
            'description': 'Selecting the best fully connected model',
            'platform': 'colab', # Executed in Google Colab
            'params': {
                'path': 'model_on_top_selection', # Path to the folder with logs and weights of the fully connected model
                'batch_size': 64, # Batch size
                'optimizer_name': OPTIMIZER, # Optimizer,
                'initial_learning_rate': MODEL_ON_TOP_INITIAL_LEARNING_RATE, # Initial learning rate
                'learning_rate_decay_rate': MODEL_ON_TOP_LEARNING_RATE_DECAY_RATE, # Learning rate decay rate
                'epochs': 100, # Number of epochs when measuring inference time
                'patience': 10, # Max. number of epochs without accuracy improvement
                'process_csv': 'model_on_top_selection.csv', # Path with model training results
                'result_csv': 'selected_model_on_top.csv', # Path to the file with the description of the selected base model
            }
        },
        {
            'name': 'model_on_top_training',
            'description': 'Training fully connected models',
            'platform': 'colab', # Runs in Google Colab
            'params': {
                'path': 'model_on_top_training', # Path to the folder with logs and weights of the fully connected model
                'flip': RANDOM_FLIP, # Random image cropping
                'rotation_factor': RANDOM_ROTATION_FACTOR, # Factor of random rotation (counter-clockwise or clockwise) of the image during augmentation, fraction of 360°
                'zoom_factor': RANDOM_ZOOM, # Factor of random zooming in or out of the image during augmentation
                'contrast_factor': RANDOM_CONTRACT_FACTOR, # Factor of random change in image contrast
                'brightness_factor': RANDOM_BRIGHTNESS_FACTOR, # Factor of random change of image brightness
                'batch_size': 32, # Batch size
                'buffer_size': 100, # Buffer size
                'optimizer_name': OPTIMIZER, # Optimizer,
                'initial_learning_rate': MODEL_ON_TOP_INITIAL_LEARNING_RATE, # Initial learning rate
                'learning_rate_decay_rate': MODEL_ON_TOP_LEARNING_RATE_DECAY_RATE, # Learning rate decay factor
                'epochs': 2, # Number of epochs when measuring inference time
                'epochs_per_run': 2, # Number of training epochs per run
                'patience': 1, # Max. number of epochs without accuracy improvement
                'process_csv': 'model_on_top_training.csv', # Path with fully connected model training results
                'result_csv': 'trained_model_on_top.csv', # Path to the file with the fully connected model evaluation
            }
        },
        {
            'name': 'model_fine_tuning',
            'description': 'Model fine tuning',
            'platform': 'colab', # Performed in Google Colab
            'params': {
                'path': 'model_fine_tuning', # Path to the folder with the fully connected model logs and weights
                'batch_size': 32, # Batch size
                'buffer_size': 100, # Buffer size
                'optimizer_name': OPTIMIZER, # Optimizer
                'initial_learning_rate': MODEL_INITIAL_LEARNING_RATE, # Initial learning rate
                'learning_rate_decay_rate': MODEL_LEARNING_RATE_DECAY_RATE, # Learning rate decay rate
                'epochs': 50, # Number of training epochs
                'epochs_per_run': 10, # Number of training epochs per run
                'patience': 10, # Max. number of epochs without accuracy improvement
                'process_csv': 'model_fine_tuning.csv', # Path to the file with the model fine-tuning process data
                'result_csv': 'model.csv', # Path to the file with the evaluation of the resulting model
            }
        },
        {
            'name': 'model_deploy_test',
            'description': 'Testing the model',
            'platform': 'local', # Executed on the local computer
            'params': {
                'path': 'model_deploy_test',
                'scale_factor': 0.209, # Scaling factor for detecting faces in an image
                'min_face_size': 128, # Minimum face size for detecting faces in an image
                'min_probability': 0.5, # Minimum probability of emotion prediction
                # 'max_error': 0.25, # Maximum error of valence-arousal prediction
                'process_csv': 'model_deploy_test.csv', # Path to the file with data of the base model selection process
                'result_csv': 'emotion_files.csv', # Path to the file with description of files of received emotion images
            }
        },
    ]
}
```

</p>
</details>

#### 3.1. Extracting features from the training dataset (`train_feature_extraction`)
The goal of this step is to obtain a dataset from the feature vectors of the training dataset's facial images. This dataset is used in [the best on top model selectionl selection](#33-selecting-the-best-top-model-model_on_top_selection) stage. Features are extracted by "passing" face images through the base model.
Images are fed to the model in batches. The batch size is specified by the `batch_size` parameter. To increase the speed of processing, images are pre-loaded into a buffer (English *buffer*), the size of which is specified using the `buffer_size` parameter.
The resulting feature vectors are packed into a numpy array, which is saved to a file with the name specified in the `path` parameter and the `npz` extension. The zip archive of this file is copied to the project folder in a single cloud storage.

#### 3.2. Extracting features from the test dataset (`test_feature_extraction`)
The goal of this step is to obtain a dataset from the feature vectors of the test dataset faces. This dataset is also used in the step of selecting the best fully connected model. This step is performed similarly to the previous one and has the same set of tuning parameters.

#### 3.3. Selecting the best on top model (`model_on_top_selection`)
The goal of this stage is to determine the top model configuration that has the best potential for use in the resulting model. To identify such a configuration, top models are trained with all possible combinations of options for the number of dropout layer blocks and fully connected layers, options for the number of output neurons in fully connected layers, as well as options for the values ​​of the proportion of zeroed neurons in dropout layers. Top models are trained on a training dataset of face image features.

Models are trained until an increase in the accuracy of their predictions is observed on the test dataset of face image features. Accuracy is checked on the [skillbox-computer-vision-project](https://www.kaggle.com/competitions/skillbox-computer-vision-project/data) platform. During the training process, its speed decreases every epoch along an exponential trajectory determined by the initial value specified by the `initial_learning_rate` parameter and the decay factor specified by the `initial_learning_rate` parameter.

The data from the model training process is saved in the folder with the name specified in the `path` configuration parameter. The `logs` subfolder contains the history files of the training metrics:
- `learning rate` - the learning rate during the next training epoch;
- `loss` - the value of the loss function at the end of the training epoch;
- `sparse_categorical_accuracy` - the accuracy of predictions on the training dataset at the end of the training epoch;
- `test_public_score` - the accuracy of predictions of the model on the public part of the test dataset;
- `test_private_score` - the accuracy of predictions of the model on the private part of the test dataset;
- `test_score` - the average accuracy of predictions of the model on the test dataset.

At the end of training each model, its file with the "best" weights (weights with which the model showed the best accuracy) is saved in the nested `models`.

The training process is visualized on the interactive panel [TensorBoard](https://www.tensorflow.org/tensorboard) built into the notebook.

The model that showed the highest best value of prediction accuracy on the test dataset is selected as the best top model.

At the end of the stage, the contents of the data folder are copied as a zip archive of the same name to the project folder in a single cloud storage.

<details><summary>Fields of the detailed information csv file about the stage execution</summary>
<p>

`model_building/model_on_top_selection.csv`:
- `model_on_top_config` - configuration of the dropout layer blocks and the fully connected layers of the top model in the format:
(`drop_out_rate_1`, `dense_units_1`), (`drop_out_rate_2`, `dense_units_2`), ..., (`drop_out_rate_n`, `dense_units_n`),
where&nbsp;`n` is the number of blocks from the list specified by the `MODEL_ON_TOP_DENSE_NUMS` parameter;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`drop_out_rate_i` (i=1...n) - the proportion of input neurons of the fully connected layer in the i-th block from the list, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; specified by the `MODEL_ON_TOP_DROPOUT_RATES` parameter, that are reset during training; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`dense_output_i` (i=1...n) - the number of output neurons of the fully connected layer in the i-th block from the list specified by the parameter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`MODEL_ON_TOP_DENSE_UNITS`;
- `best_epoch` - the number of the training epoch at the end of which the best prediction accuracy was shown on the test dataset;
- `loss_at_best_epoch` - the value of the loss function at the end of the training epoch at the end of which the best prediction accuracy was shown on the test dataset;
- `sparse_categorical_accuracy_at_best_epoch` - the accuracy value on the training dataset at the end of the training epoch at the end of which the best prediction accuracy was shown on the test dataset;
- `best_test_score` - the best prediction accuracy value on the test dataset.

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`model_building/selected_model_on_top.csv`:
- `model_on_top_config` - configuration of the units of the dropout and fully connected layers of the selected top model in the format:
(`drop_out_rate_1`, `dense_units_1`), (`drop_out_rate_2`, `dense_units_2`), ..., (`drop_out_rate_n`, `dense_units_n`),
where&nbsp;`n` is the number of units from the list specified by the `MODEL_ON_TOP_DENSE_NUMS` parameter;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`drop_out_rate_i` (i=1...n) - the proportion of input neurons of the fully connected layer in the i-th block from the list, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; specified by the `MODEL_ON_TOP_DROPOUT_RATES` parameter, that are reset during training; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`dense_output_i` (i=1...n) - the number of output neurons of the fully connected layer in the i-th block from the list specified by the parameter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`MODEL_ON_TOP_DENSE_UNITS`;
- `best_test_score` - the best value of prediction accuracy on the test dataset of the selected upper model.

</p>
</details>

<details><summary>Example of model training graphs</summary>
    <p align="center" style="text-align:center">
        <table>
            <tr>
                <td>
                    <p align="center"><b>learning rate</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170698-ac024401-65f8-4583-ae99-cbbf797ec7e1.svg>
                </td>
                <td>
                    <p align="center"><b>train loss</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170699-00e42311-c1c5-4038-8494-be6eda9d24f0.svg>
                </td>
                <td>
                    <p align="center"><b>train accuracy</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170697-5901a5bd-27d5-47b8-96a8-87783a184d6d.svg>
                </td>
            </tr>
            <tr>
                <td>
                    <p align="center"><b>test public score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170696-372b2f6f-8cde-4d56-b39a-db42b02fd6f0.svg>
                </td>
                <td>
                    <p align="center"><b>test private score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170692-644e7ac6-7a01-4226-a448-f3c5fd2364ba.svg>
                </td>
                <td>
                    <p align="center"><b>test score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170691-85d0656e-82d3-4ce9-a4f8-d6bc419cb1a1.svg>
                </td>
            </tr>
        </table>
    </p>
</details>

#### 3.4. Training a fully connected model (`model_on_top_training`)
At this stage, the top model is trained. The purpose of such training is primarily to reduce the likelihood of an excessively large gradient during training of the entire model, which can lead to the "destruction" of the base model. But on the other hand, training only the top model for too long can complicate subsequent training of the entire model. Therefore, it is recommended to train a fully connected model only for a few epochs. The maximum number of training epochs is determined by the `epochs` parameter. However, training may end earlier if the accuracy of the model's predictions on test data stops growing (does not increase over a certain number of epochs specified by the `patience` parameter).

To increase the diversity of training data, training face images are preliminarily randomly transformed using an augmentation model.

Any optimizer from the [Keras](https://keras.io/api/optimizers/#available-optimizers) library can be used for training. The optimizer name must be specified in the `optimizer_name` parameter. During training, its speed is reduced every epoch along an exponential trajectory determined by the initial value specified by the `initial_learning_rate` parameter and the reduction factor specified by the `initial_learning_rate` parameter.

Training is performed in batches. The batch size is determined by the `batch_size` parameter. To increase the speed of processing, training face images are pre-loaded into a buffer, the size of which is specified by the `buffer_size` parameter.

Training data is saved in a folder named by the `path` parameter. The `logs` subfolder contains training metrics history files:
- `learning rate` - the learning rate during the next training epoch;
- `loss` - the value of the loss function at the end of the training epoch;
- `sparse_categorical_accuracy` - the accuracy of predictions on the training dataset of face images at the end of the training epoch;
- `test_public_score` - the accuracy of the model's prediction on the public part of the test dataset of face images;
- `test_private_score` - the accuracy of the model's prediction on the private part of the test dataset of face images;
- `test_score` - the average accuracy of the model's prediction on the test dataset of face images.

The stage can be divided into several iterations, the number of which is determined by the `epochs_per_run` parameter. At the end of each iteration or when training stops, the model with the latest and "best" weights (weights with which the model showed the best accuracy) are saved in the `models` nested folder. At the beginning of the next iteration, the model is not created anew, but is loaded from a file with the latest weights of the previous iteration.

The training process is visualized on the interactive panel [TensorBoard](https://www.tensorflow.org/tensorboard) built into the notebook.

Once the stage is completed, the contents of the data folder are copied as a zip archive of the same name to the project folder in a single cloud storage.

<details><summary>Fields of the stage execution report csv file</summary>
<p>

`model_building/model_on_top_training.csv`:
- `epoch` - training epoch number;
- `loss` - loss function value at the end of the training epoch;
- `sparse_categorical_accuracy` - prediction accuracy on the training face dataset at the end of the training epoch;
- `lr` - learning rate during the next training epoch;
- `test_public_score` - model prediction accuracy on the public part of the test face dataset;
- `test_private_score` - model prediction accuracy on the private part of the test face dataset;
- `test_score` - average prediction accuracy on the test face dataset.

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`model_building/trained_model_on_top.csv`
- `base_model_name` - the name of the base model;
- `model_on_top_config` - configuration of the units of the dropout layers and fully connected layers of the top model in the format:
(`drop_out_rate_1`, `dense_units_1`), (`drop_out_rate_2`, `dense_units_2`), ..., (`drop_out_rate_n`, `dense_units_n`),
where&nbsp;`n` is the number of units from the list specified by the `MODEL_ON_TOP_DENSE_NUMS` parameter;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`drop_out_rate_i` (i=1...n) - the proportion of input neurons of the fully connected layer in the i-th block from the list, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; specified by the `MODEL_ON_TOP_DROPOUT_RATES` parameter, that are reset during training; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`dense_output_i` (i=1...n) - the number of output neurons of the fully connected layer in the i-th block from the list specified by the parameter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`MODEL_ON_TOP_DENSE_UNITS`;
- `best_test_score` - the best value of the model's prediction accuracy on the test dataset.

</p>
</details>

<details><summary>Example of model training graphs</summary>
    <p align="center" style="text-align:center">
        <table>
            <tr>
                <td>
                    <p align="center"><b>learning rate</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170501-a44927a4-a898-4ef9-b5ec-3418a5bf8ebd.svg>
                </td>
                <td>
                    <p align="center"><b>train loss</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170473-72fe76b7-074c-43c0-8925-5152b7d3be7b.svg>
                </td>
                <td>
                    <p align="center"><b>train accuracy</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170479-03e77aaf-d219-4211-8acd-72016b88fc36.svg>
                </td>
            </tr>
            <tr>
                <td>
                    <p align="center"><b>test public score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170477-0dcea241-f0a0-4f54-a999-e9b049145032.svg>
                </td>
                <td>
                    <p align="center"><b>test private score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170478-dabd1a34-3185-4c14-b308-0266470748d1.svg>
                </td>
                <td>
                    <p align="center"><b>test score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200170479-03e77aaf-d219-4211-8acd-72016b88fc36.svg>
                </td>
            </tr>
        </table>
    </p>
</details>

#### 3.5. Model fine-tuning (`model_fine_tuning`)

Model fine-tuning is the final stage of model training. During this stage, both the upper and base models are trained simultaneously. At the beginning of the stage, the model is loaded from a file with the "best" weights obtained in the previous stage. Otherwise, the process of performing this stage is similar to the previous one and is also visualized using the built-in interactive panel [TensorBoard](https://www.tensorflow.org/tensorboard). The stage has the same tuning parameters.

After training is complete, the contents of the data folder (metric change history files and the latest and "best" model files) are copied as a zip archive of the same name to the project folder in a single cloud storage.

<details><summary>Fields of the stage execution report csv file</summary>
<p>

`model_building/model_fine_tuning.csv`:
- `epoch` - training epoch number;
- `loss` - loss function value at the end of the training epoch;
- `sparse_categorical_accuracy` - prediction accuracy on the training face dataset at the end of the training epoch;
- `lr` - learning rate during the next training epoch;
- `test_public_score` - model prediction accuracy on the public part of the test face dataset;
- `test_private_score` - model prediction accuracy on the private part of the test face dataset;
- `test_score` - average prediction accuracy on the test face dataset.

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`model_building/model.csv`:
- `base_model_name` - the name of the base model;
- `model_on_top_config` - configuration of the dropout layer blocks and fully connected layers of the top model in the format:
(`drop_out_rate_1`, `dense_units_1`), (`drop_out_rate_2`, `dense_units_2`), ..., (`drop_out_rate_n`, `dense_units_n`),
where&nbsp;`n` is the number of blocks from the list specified by the `MODEL_ON_TOP_DENSE_NUMS` parameter;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`drop_out_rate_i` (i=1...n) - the proportion of input neurons of the fully connected layer in the i-th block from the list, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; specified by the `MODEL_ON_TOP_DROPOUT_RATES` parameter, that are reset during training; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`dense_output_i` (i=1...n) - the number of output neurons of the fully connected layer in the i-th block from the list specified by the parameter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`MODEL_ON_TOP_DENSE_UNITS`;
- `best_test_score` - the best value of the model's prediction accuracy on the test dataset.

</p>
</details>

<details><summary>Example of model training graphs</summary>
    <p align="center">
        <table>
            <tr>
                <td>
                    <p align="center"><b>learning rate</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200162240-69f4a03f-af7e-486d-b595-3f939339dc95.svg>
                </td>
                <td>
                    <p align="center"><b>train loss</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200162241-0b956ee6-865d-46c4-b08c-38bf741f724b.svg>
                </td>
                <td>
                    <p align="center"><b>train accuracy</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200162239-91d315bb-b643-4827-989b-260e99879b2b.svg>
                </td>
                </tr>
            <tr>
                <td>
                    <p align="center"><b>test public score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200162238-31226353-fb24-4fa6-a855-2dbf3f531e37.svg>
                </td>
                <td>
                    <p align="center"><b>test private score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200162237-9004ca30-26a0-4f29-99c0-8fbf5272491d.svg>
                </td>
                <td>
                    <p align="center"><b>test score</b></p>
                    <img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/200162235-4a95bdf5-2bee-4e6e-82d1-951092ea14b4.svg>
                </td>
            </tr>
        </table>
    </p>
</details>

#### 3.6. Model testing (`model_deploy_test`)

Testing is performed on an image from the built-in camera of the local computer. Therefore, this stage must be performed on the local computer (the `platform` parameter must be equal to `local`).

The presence and coordinates of a face in the image are determined using the [MTCNN](https://github.com/ipazc/mtcnn) face detector. The emotion that the face expresses is predicted using an instance of the `FaceEmotionRecognitionNet` class based on the created model with the "best" weights.

The image from the camera is displayed in a separate window, which is automatically opened when the stage is performed and closed upon its completion. The face area in the image is highlighted using a rectangle. The remaining visual information depends on the model type.

##### 3.6.1. Testing the [1st type](#1-a-model-that-predicts-the-probabilities-of-emotions-that-a-person-is-experiencing-based-on-their-facial-expression) model.
If the probability of recognizing an emotion exceeds the specified reliability threshold, defined by the `min_probability` parameter, the face area rectangle is colored green. Otherwise, this rectangle is colored red.

The name of the recognized emotion and its probability in brackets are displayed above the face area rectangle on a background of the same color.

On the left side of the window, a list of emotions that were recognized with a high probability (exceeding the reliability threshold) is displayed in green. The maximum probability of their recognition is displayed next to the name of the emotions in brackets. On the right side, on the contrary, a list of emotions that have not yet been reliably recognized is displayed. The maximum probability of their recognition is also displayed next to the names of these emotions in brackets.

You can finish testing after all emotions have been reliably recognized (this is indicated by the appearance of the inscription "All emotions have been recognized. Press any key to complete." in the window). By clicking the button, the output window closes.

After this, the facial images that most reliably express the recognizable emotions are saved in a folder with the name specified by the `path` parameter. A copy of these images in the form of a zip archive with the same name is copied to the project folder in a single cloud storage. The paths to the facial image files and the probability of emotion recognition are saved in the csv file of the results in the project folder in a single cloud storage.

<details><summary>Stage completion report csv file fields</summary>
<p>

`model_building/model_deploy_test.csv`:
- `emotion` - emotion name;
- `image` - face image;
- `probability` - probability with which the emotion was recognized.

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`model_building/emotion_files.csv`:
- `emotion` - the name of the emotion;
- `image` - the path to the face image file;
- `probability` - the probability with which the emotion was recognized.

</p>
</details>

<details><summary>Examples of window appearance</summary>
<p align="center">
<table>
<tr>
<td>
<p align="center"><b>During testing</b></p>
<img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/204358136-1178be9e-b96d-4722-9058-06592098d9cf.png>
</td>
<td>
<p align="center"><b>Upon completion of testing</b></p>
<img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/204358122-9b2fc697-a512-4bf5-835a-de7cd7ae564c.png>
</td>
</tr>
</table>
</p>
</details>

<details><summary>Examples of the “best” facial images</summary>
<p align="center">
<img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/204360816-84a96f46-cdb8-45fc-9560-5c1b01a01be5.png>
</p>
</details>

##### 3.6.2. Testing the [2nd type](#2-a-model-that-can-recognize-the-valence-and-intensity-levels-of-an-emotion-that-a-person-is-experiencing-on-a-scale-from--1-to-1-based-on-their-facial-expression) model.
Above the rectangles of the face area in the first line the valence values ​​are displayed ( `V`) and intensity (`A`) of emotion. And in the second line the name of the emotion is indicated, they are closest to the typical values ​​of valence and intensity. The distance is indicated in brackets
If the distance between the typical and model-defined values ​​of valence and intensity of emotion does not exceed the specified reliability threshold, defined by the parameter `max_error`, then the face area rectangle and the text background are colored green. Otherwise, the rectangle and text are colored red color.

On the left side of the window, a list of emotions that were reliably recognized is displayed in green (the distance of the closest obtained valence and intensity values ​​to the typical valence and intensity values ​​does not exceed the reliability threshold). Next to the name of the emotions, the value of the "best" valence values ​​is displayed in brackets and intensity. On the right side, on the contrary, a list of emotions is displayed that have not yet been reliably recognized. Next to the names of these emotions, the value of the "best" valence and intensity values ​​is also displayed in brackets.

Just as in the case of testing the 1st type model, testing can be completed by the user only after all emotions have been reliably recognized. After this, the "best" facial images are also saved in the project folder on the execution platform and in a single cloud storage.

<details><summary>Stage execution report csv file fields</summary>
<p>

`model_building/model_deploy_test.csv`:
- `emotion` - emotion name;
- `image` - face image;
- `error` - distance between typical and recognized values ​​of valence and intensity of emotion;
- `valence` - valence of recognized emotion;
- `arousal` - intensity of recognized emotion.

</p>
</details>

<details><summary>Fields of the csv file of the stage execution results</summary>
<p>

`model_building/emotion_files.csv`:
- `emotion` - the name of the emotion;
- `image` - the path to the face image file;
- `error` - the distance between typical and recognized values ​​of valence and intensity of emotion;
- `valence` - the valence of the recognized emotion;
- `arousal` - the intensity of the recognized emotion.

</p>
</details>

<details><summary>Examples of window appearance</summary>
<p align="center">
<table>
<tr>
<td>
<p align="center"><b>Under testing</b></p>
<img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/204358144-a0d4f282-84d9-4879-9b47-1fa3f4709131.png>
</td>
<td>
<p align="center"><b>Upon completion of testing</b></p>
<img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/204358140-37ff0cf8-7f70-4a75-966b-1aa2aaf9300c.png>
</td>
</tr>
</table>
</p>
</details>

<details><summary>Examples of the “best” facial images</summary>
<p align="center">
<img align="center" width=100% src=https://user-images.githubusercontent.com/107345313/204360810-2e86e4c3-e5bd-4fbe-8dbf-5d9ecf614c8c.png>
</p>
</details>

## Usage

To use the notebook, simply complete the following steps:

1. Prepare an empty storage on the [Google Drive](https://drive.google.com/) platform.
2. Install the [Google Drive](https://drive.google.com/) application on your computer and connect the prepared storage as a logical drive on your local computer.
3. Register on the [Google Colab](https://colab.research.google.com/) platform under the same user as on the [Google Drive](https://drive.google.com/) platform.
4. Register on the [Kaggle](https://www.kaggle.com) platform.
5. Take part in the [skillbox-computer-vision-project](https://www.kaggle.com/c/skillbox-computer-vision-project) competition.
6. Create an API Token in your profile settings on the [Kaggle](https://www.kaggle.com) platform and copy it to the [Google Drive](https://drive.google.com/) cloud storage.
7. Open the notebook remotely in [Google Colab](https://colab.research.google.com/) or on your local computer in [JupyterNotebook](https://jupyter.org/) or [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html).
8. In the project settings, specify a link to the API Token in the [Google Drive](https://drive.google.com/) cloud storage.
9. Complete the basic settings (a brief purpose of the settings is given in the comments to them, and a more detailed description is given above in the text of this document).
10. In the settings of each stage, specify on which platform it will be executed.

## Directions for further development

Although the notebook was developed to solve a specific problem, it can be easily adapted to solve any other image classification problem. To do this, it is necessary to replace the mechanism for obtaining the accuracy of the model's predictions on test data (the `Kaggle` class) and use its instance in the early stopping class in the absence of an increase in the accuracy of predictions `EarlyStoppingAtMaxTestScore`.
