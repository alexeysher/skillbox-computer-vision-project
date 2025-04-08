import streamlit as st

st.markdown(
    f'''
    ## Introduction
    
    This work was carried out as part of the 
    [skillbox-computer-vision-project](https://www.kaggle.com/c/skillbox-computer-vision-project) competition 
    on the [Kaggle](https://www.kaggle.com) platform.
    Within the thesis, two types of a ready-to-use models for recognizing human emotions based on facial expressions
    were developed:
    
    1. A [model](#1st-type) that predicts the probabilities of emotions:
        - anger
        - contempt
        - disgust
        - fear
        - happy
        - neutral
        - sad
        - surprise
        - uncertain
    2. A [model](#2nd-type) that recognizes the valence (the extent to which an emotion is positive or negative) 
    and arousal (refers an emotion intensity, i.e., the strength of the associated emotional state).
    Both the metrics have scale from -1 to +1.
    
    Within the work the [notebooks](https://github.com/alexeysher/skillbox-computer-vision-project) were developed. 
    The models were developed using [Jupyter Notebook]() and [Google Colab](https://colab.research.google.com/) tools. 
    
    The models were created using [Keras](https://keras.io/api/) and 
    [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) frameworks.
    The [Transfer Learning](https://keras.io/guides/transfer_learning/) and 
    [Fine Tuning](https://keras.io/guides/transfer_learning/) approaches were applied.

    The model creation process is divided into three independent parts (pipelines), 
    which should be performed sequentially:
    1. [Base model selection](base_model_selection#base-model-selection)
    2. [Image preprocessing](image_preprocessing#image-preprocessing)
    3. [Model building](model_building#model-building)
    
    The both developed models shows very good evaluation results on the 
    [test data](http://localhost:8501/provided_data#test-data).
    They got `0.5146` and `0.4846` scores, respectively, while the required one is `0.4`.
    
    The ready-to-user models were deployed on [Google Cloud Vertex AI Platform](https://cloud.google.com/vertex-ai).
    
    Two demos usage of them are provided:
    - [Processing image](camera#camera) from camera
    - [Creating trailer](video#video) for one person video
    '''
)