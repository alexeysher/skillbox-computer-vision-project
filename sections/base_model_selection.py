import pandas as pd
import streamlit as st
from auxiliary import DEFAULT_COLORS

st.markdown(
    '''
    # Base model selection
    '''
)

st.markdown(
    '''
    ## Filtering base models by size

    The goal of this stage is to exclude from the list of Base Models too large base models 
    which are not applicable for using in the developing models.
    '''
)

with st.expander('See details...'):


    st.markdown('### Settings')
    max_size = 64

    df = pd.DataFrame(
        [
            [
                'Max. size',
                'Maximal size of applicable Base Models (MB)',
                max_size,
            ],
        ],
        columns=['Parameter', 'Description', 'Value']
    ).set_index('Parameter')

    st.dataframe(
        df,
        use_container_width=False
    )

    st.markdown(
        '''
        ### Applicable Base Models
        '''
    )
    df = st.session_state.base_models
    df = df.loc[df['size'] <= max_size]
    df_style = df.style
    # df_style.set_properties(
    #     **{'background-color': DEFAULT_COLORS[8]},
    #     subset=pd.IndexSlice[df['size'] > 64, 'size']
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
    st.dataframe(
        df_style,
        column_config=config,
        use_container_width=False,
        height=(df.shape[0] + 1) * 35 + 2
    )

# st.markdown(
#     '''
#     ## Collecting information about sizes of input images and feature vectors
#
#     The first goal of this stage is to *obtain information about the optimal sizes of input images*
#     for each available base model [Keras Applications](https://keras.io/api/applications/).
#     Images fed to the model during training and use for obtaining predictions will be scaled to this size.
#
#     The second goal is to obtain information about the sizes of feature vectors at the output
#     of the last pooling layer of each base model. This information is needed to create the model on top model,
#     which provides the transformation of the feature vector, depending on the model type,
#     either into a vector of emotion probabilities or into a pair of valence and intensity values.
#     '''
# )
#
# with st.expander('See details...'):
#     st.markdown('### Collected information')
#     df = st.session_state.base_model_sizes
#     df_style = df.style
#     config = {
#         'base_model_name': 'Model',
#         'image_size': 'Image Size',
#         'feature_size': 'Feature Size'
#     }
#     st.dataframe(
#         df_style,
#         column_config=config,
#         height=(df.shape[0] + 1) * 35 + 2,
#         use_container_width=False
#     )

st.markdown(
    '''
    ## Measuring the inference time of models
    
    The goal of this step is to estimate the inference time of a model consisting of a base model combined 
    with the "heaviest" model on top (the model containing the maximum number of fully connected layers 
    of the maximum size). This information allows us to exclude base models with insufficient performance 
    from further consideration.
    '''
)

with (st.expander('See details...')):
    st.markdown('### Settings')

    df = pd.DataFrame(
        [
            ['Model on top', 'The "heaviest" model on top', '(0, 1024), (0, 1024)', '(0, 1024), (0, 1024)'],
            ['Repetitions', 'Number of repetitions', '100', '100'],
        ],
        columns=['Parameter', 'Description', '1st type', '2nd type']
    ).set_index('Parameter')

    st.dataframe(
        df,
        use_container_width=False
    )

    df = st.session_state.base_model_inference_times
    df = df.droplevel(1, axis=1)
    df_style = df.style
    # df.highlight_min(
    #     # subset=[(1, 'inference_time'), (2, 'inference_time')],
    #     props=f'color: white; background-color: {DEFAULT_COLORS[1]};'
    # ).highlight_max(
    #     # subset=[(1, 'inference_time'), (2, 'inference_time')],
    #     props=f'color: white; background-color: {DEFAULT_COLORS[0]};'
    # )
    # df_style = df.style.background_gradient(
    #     # subset=[(1, 'inference_time'), (2, 'inference_time')],
    #     # props=f'color: white; background-color: {DEFAULT_COLORS[1]};'
    # ).highlight_max(
    #     # subset=[(1, 'inference_time'), (2, 'inference_time')],
    #     props=f'color: white; background-color: {DEFAULT_COLORS[0]};'
    # )
    config = {
        'base_model_name': 'Model',
        1: st.column_config.ProgressColumn(
            '1st type',
            min_value=df[1].min(), max_value=df[1].max(),
            format='%d'
        ),
        2: st.column_config.ProgressColumn(
            '2nd type',
            min_value=df[2].min(), max_value=df[2].max(),
            format='%d'
        ),
    }
    st.markdown('### Measured inference times (ms)')
    st.dataframe(
        df_style,
        column_config=config,
        height=(df.shape[0] + 1) * 35 + 2,
        use_container_width=False
    )

st.markdown(
    '''
    ## Searching for the most suitable base model
    
    The goal of this stage is to find a base model that will be used to build the resulting model. 
    The selection is based on the best combination of performance and accuracy of the model shown on the 
    [ImageNet](https://www.image-net.org/) validation dataset, taking into account the weights 
    of these metrics.
    '''
)

with st.expander('See details...'):
    st.markdown(
        '''
        ### Settings
        '''
    )

    df = pd.DataFrame(
        [
            ['Max. inference time', 'Maximum allowed model inference time (ms)', 330, 330],
            ['Accuracy weight', 'Weight of Top-1 Accuracy', 0.4, 0.6],
            ['Inference Time weight', 'Weight of Inference Time', 0.6, 0.4],
        ],
        columns=['Parameter', 'Description', '1st type', '2nd type']
    ).set_index('Parameter')

    st.dataframe(
        df,
        use_container_width=False
    )

    st.markdown(
        '''            
        ### Models ranking
        '''
    )
    st.markdown(
        '''
        For each selected model, points are awarded for accuracy and for performance. 
        Accuracy scores $s_a$ are awarded according to the following rule:
        '''
    )
    st.latex(
        '''
        {s_a}_i = {a_i - a_{min}\over a_{max} - a_{min}}
        '''
    )
    st.markdown(
        '''
        where $a_i$ - the accuracy of the estimated $i$th model, 
        $a_{min}, a_{max}$ - the minimal and maximal value of the accuracy through the all estimated models.
        '''
    )

    st.markdown(
        '''        
        Performance scores $s_t$ are awarded according to the following rule:
        '''
    )

    st.latex(
        r'''
        {s_t}_i = {t_{max} - t_i \over t_{max} - t_{min}}
        '''
    )

    st.markdown(
        '''
        where $t_i$ - the inference time of the estimated $i$th model, 
        $t_{min}, t_{max}$ - the minimal and maximal value of the inference time through the all estimated models.
        '''
    )

    st.markdown(
        '''
        The total weighted scores for accuracy and performance take into account the ratio of the importance 
        of accuracy and performance of the model and are determined by the following rule:
        '''
    )

    st.latex(
        r'''
        s_i = {s_a}_i \omega_a + {s_t}_i \omega_t
        '''
    )

    st.markdown(
        '''
        where $\omega_a$ and $\omega_t$ are the weights of the accuracy and the performance respectively.
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        The obtained scores and ranks for the base models are listed below:
        ''',
        unsafe_allow_html=True
    )

    df = st.session_state.base_model_scores
    df.rename({1: '1st type', 2: '2nd type'}, axis=1, inplace=True)
    # df.columns = pd.MultiIndex.from_tuples(
    #     [
    #         ('1st type', 'inference_time_score'),
    #         ('1st type', 'weighted_score'),
    #         ('1st type', 'rank'),
    #         ('2nd type', 'inference_time_score'),
    #         ('2nd type', 'weighted_score'),
    #         ('2nd type', 'rank'),
    #     ]
    # )
    df_style = df.style
    # df_style.highlight_min(
    #     subset=[('1st type', 'rank'), ('2nd type', 'rank')],
    #     props=f'color: white; background-color: {DEFAULT_COLORS[8]};'
    # )
    config = {
        'base_model_name': 'Model',
        'top1_accuracy_score': st.column_config.NumberColumn('Accuracy Score', format='%.3f'),
        'inference_time_score': st.column_config.NumberColumn('Inference Time Score', format='%.3f'),
        'weighted_score': st.column_config.ProgressColumn(
            'Weighted Score', min_value=0, max_value=1, format='%.3f'
        ),
        'rank': 'Rank'
    }
    st.dataframe(
        df_style,
        column_config=config,
        height=(df.shape[0] + 2) * 35 + 2,
        use_container_width=False
    )

st.markdown(
    '''            
    ## Results

    For further work, the models with rank #1 were selected, 
    i.e. the models witch had the best weighted score for accuracy and performance:
    - [EfficientNetB0](https://keras.io/2.18/api/applications/efficientnet/) - for the [1st type](intro#1st-type)
    - [EfficientNetB2](https://keras.io/2.18/api/applications/efficientnet/) - for the [2nd type](intro#2nd-type)
    '''
)

st.markdown(
    '''
    '''
)

