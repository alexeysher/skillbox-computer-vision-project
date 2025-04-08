import pandas as pd
import streamlit as st
from auxiliary import DEFAULT_COLORS


st.markdown(
    '''
    # Models architecture
    
    As mentioned earlier, the models were created using the [Transfer Learning](https://keras.io/guides/transfer_learning/) and 
    [Fine Tuning](https://keras.io/guides/transfer_learning/) approaches. 
    That is, the models were not created from scratch, but based on a pretrained models, 
    the so-called *base models*. Any model from the [Keras Applications](https://keras.io/api/applications/) 
    library trained on the [ImageNet](https://www.image-net.org/) could serve as a base model:
    '''
)

df = st.session_state.base_models
df_style = df.style
# df_style.highlight_min(
#     subset='size',
#     props=f'color: white; background-color: {DEFAULT_COLORS[1]};'
# ).highlight_min(
#     subset='accuracy',
#     props=f'color: white; background-color: {DEFAULT_COLORS[0]};'
# ).highlight_max(
#     subset='size',
#     props=f'color: white; background-color: {DEFAULT_COLORS[0]};'
# ).highlight_max(
#     subset='accuracy',
#     props=f'color: white; background-color: {DEFAULT_COLORS[1]};'
# )
config = {
    'name': 'Model',
    'size': st.column_config.ProgressColumn(
        'Size (MB)',
        min_value=df['size'].astype(float).min(), max_value=df['size'].astype(float).max(),
        format='%d'
    ),
    'accuracy': st.column_config.ProgressColumn(
        'Top-1 Accuracy (%)',
        min_value=df['accuracy'].min(), max_value=df['accuracy'].max(),
        format='%.1f'
    )
}
with st.expander('See list of available Base Models...'):
    st.dataframe(
        df_style,
        column_config=config,
        height=(st.session_state.base_models.shape[0] + 1) * 35 + 2,
        use_container_width=False
    )

st.markdown(
    '''
    The particular base models were selected based on the accuracy of its predictions 
    on the [ImageNet](https://www.image-net.org/) validation dataset and the measured inference speed 
    of the resulting model built on the basis of this base model (for more details, see the 
    [Base Model Selection](/base_model_selection#base-model-selection).
     
    The base model is trained to determine which object from the 
    [ImageNet](https://www.image-net.org/) dataset is presented in the input image based on its features. 
    The convolutional part of the model is responsible for feature extraction. 
    The last, fully connected layer (*dense layer*) with the 
    [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) activation function, 
    located as if on top of the model, is responsible for identifying the object based on its features. 
    Since the model being built solves a completely different problem, 
    this fully connected layer is not needed in the resulting model. 
    Therefore, only the convolutional part of the base model is used. 
    At the top of the convolutional part is the pooling layer. 

    To identify emotions based on the features obtained using the base model, 
    one or more fully connected layers with a [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) 
    type activation function can be added on top of it. 
    In addition, to ensure regularization during the training process, a dropout layer is placed before each such layer. 
    During the training process, such a layer randomly zeroes a given portion of the input neurons. At the same time, 
    the remaining input neurons are proportionally increased so that the sum of all input neurons does not change.
    
    At the top of [1st type](#1st-type) 
    model another fully connected layer with the function was added to the last block activation of the 
    [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) model and the number of output neurons corresponding 
    to the number of recognized emotions. In this case, at the output of the model, during inference, 
    a vector of probabilities is formed for what emotion the person's face expresses at the input image.
    
    At the top of [2nd type](#2nd-type) 
    model another fully connected layer is also added to the last block. 
    But this layer has only 2 output neurons, and the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) 
    function is used for activation, with a value limit of up to 2. 
    Thus, the output neurons of this layer can take values in the range from 0 to 2. 
    But since the predicted values of valence and intensity must be in the range from -1 to 1, 
    then at the end of the model another layer was placed, which subtracts 1 from the values of 
    the output neurons of the previous fully connected layer.
    
    All the layers added to output of the base model from so-called the "model on top".
    During the model building the optimal configuration through the 
    '''
)

with st.expander('See list of Models On Top...'):
    df = st.session_state.models_on_top
    df.index.name = 'Model'
    df_style = df.style.format(
        subset=[('Block #1', 'Dropout Rate'), ('Block #2', 'Dropout Rate')],
        precision=1
    ).format(
        subset=[('Block #1', 'Dense Units'), ('Block #2', 'Dense Units')],
        precision=0
    )
    st.dataframe(
        df_style,
        height=(df.shape[0] + 2) * 35 + 2,
        column_config=config,
        use_container_width=False
    )


st.markdown(
    '''
    Additionally in order to improve the quality of model training by increasing the diversity of input images, 
    an augmentation model is added to the base model. This model randomly transforms 
    the input image during training before feeding it to the base model. 
    The augmentation model randomly rotates the image slightly, changes its contrast and brightness, 
    and also mirrors it.
    '''
)

c1, c2 = st.columns(2, gap='medium')
with c1:
    st.markdown('1st type')
    st.image('https://user-images.githubusercontent.com/107345313/'
             '200283993-4ec70b6c-9da7-4355-891a-564332559041.svg',
             use_container_width=True)
with c2:
    st.markdown('2nd type')
    st.image('https://user-images.githubusercontent.com/107345313/'
             '200284152-5faabe63-43ab-4593-828b-2684ef35b70f.svg',
             use_container_width=True)

st.markdown(
    '''
    As mentioned above the augmentation model and the dropout layers of the model on top
    works only during training time. So for inference time structure of the models
    can be simplified like shown below:
    '''
)
c1, c2 = st.columns(2, gap='medium')
with c1:
    st.markdown('1st type')
    st.image('https://user-images.githubusercontent.com/107345313/'
             '200292114-4c6a4e79-a151-463c-a3d9-e22bfa526efa.svg',
             use_container_width=True)
with c2:
    st.markdown('2nd type')
    st.image('https://user-images.githubusercontent.com/107345313/'
             '200292181-72ec54e2-f367-4c06-854f-ce541722d108.svg',
             use_container_width=True)
