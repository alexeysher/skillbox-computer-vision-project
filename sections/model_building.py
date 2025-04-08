import pandas as pd
import streamlit as st
import plotly.express as px
from auxiliary import DEFAULT_COLORS

st.markdown(
    '''
    # Model building

    ## Selecting the best model on top

    The goal of this stage is to determine the configuration of model on top that has the best potential 
    for use in the model. To identify such a configuration, models on top
    with all possible combinations (of options for the number of dropout layer blocks and fully connected layers, 
    options for the number of output neurons in fully connected layers, as well as options for the values 
    of the proportion of zeroed neurons in dropout layers) were trained and tested. 
    The features datasets were used to do it.
    
    Models had been training while an improvement in their metric was observed. Accuracy were checked on the 
    [skillbox-computer-vision-project](https://www.kaggle.com/competitions/skillbox-computer-vision-project/data) 
    platform.
    
    During the training process, the optimizer rate decreased every epoch along an exponential trajectory 
    determined by specified the initial value and the decay factor.
    
    The configurations that showed the highest value of prediction accuracy on the test dataset 
    were selected as the best models on top.

    '''
)

with (st.expander('See details...')):
    st.markdown(
        '''
        ### Training settings
        '''
    )
    df = pd.DataFrame(
        [
            ['Optimizer', 'Kind of the learning algorithm', 'Adam', 'Adam'],
            ['Initial Rate', 'Initial value of learning rate of the optimizer', 1e-4, 1e-5],
            ['Rate decay', 'Decay rate of learning rate of the optimizer', 0.96, 0.96],
            ['Loss', 'Objective that the model will try to minimize', 'Sparse Categorical Crossentropy', 'Mean Absolute Error'],
            ['Metric', 'Metric for model''s prediction quality estimation', 'Sparse Categorical Accuracy', 'Mean Absolute Percentage Error'],
            ['Epochs', 'Number of epochs to train the model', 20, 20],
            ['Patience', 'Maximum number of epochs without metric improvement', 3, 3],
        ],
        columns=['Parameter', 'Description', '1st type', '2nd type']
    ).set_index('Parameter')

    st.dataframe(
        df,
        use_container_width=False
    )

    st.markdown(
        '''
        ### Training charts
        '''
    )

    rows = []

    st.markdown(
        '''
        #### 1st type 
        '''
    )

    rows.append(st.empty())

    st.markdown(
        '''
        #### 2nd type 
        '''
    )

    rows.append(st.empty())

    losses = ['Sparse Categorical Crossentropy', 'Mean Absolute Error']
    metrics = ['Sparse Categorical Accuracy', 'Mean Absolute Error Percentage']

    for index, (row, loss, metric) in enumerate(zip(rows, losses, metrics)):
        with row:
            df = st.session_state.model_on_top_selection_logs.loc[:, index + 1]
            df.rename(
                columns={
                    'learning_rate': 'Learning Rate',
                    'loss': loss,
                    'metric': metric,
                },
                inplace=True
            )
            df.index.rename({'model': 'Model', 'epoch': 'Epoch'}, inplace=True)
            df.columns.name = 'Tag'
            stacked_df = df.stack(future_stack=True).reset_index().rename(columns={0: 'Value'})
            stacked_df.info()

            fig = px.line(
                stacked_df, x='Epoch', y='Value', color='Model', facet_col='Tag', facet_col_wrap=3,
                facet_col_spacing=0.075, facet_row_spacing=0.125,
                # height=600,
                # width=1000
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_yaxes(matches=None, visible=True, showticklabels=True)
            fig.update_traces(visible='legendonly')
            fig.update_traces(
                selector={'name': st.session_state.model_on_top_selection[(index + 1, 'score')].idxmax()},
                visible=True
            )
            fig.update_layout(margin_t=60, margin_b=60)
            st.plotly_chart(fig, use_container_width=False)

    st.markdown('### Test scores')
    df = st.session_state.model_on_top_selection
    columns = pd.MultiIndex.from_tuples(
        [
            ('1st type', 'best_epoch'),
            ('1st type', 'score_1'),
            ('2nd type', 'best_epoch'),
            ('2nd type', 'score_2')
        ]
    )
    df.columns = columns
    df_style = df.style
    df_style.highlight_max(
        subset=[('1st type', 'score_1'), ('2nd type', 'score_2')],
        props=f'color: white; background-color: {DEFAULT_COLORS[8]};'
    )
    config = {
        'model_on_top_config': 'Model',
        'best_epoch': 'Best epoch',
        # 'loss_at_best_epoch': 'Loss',
        'score_1': st.column_config.ProgressColumn(
            'Test score',
            min_value=df[[('1st type', 'score_1'), ('1st type', 'score_1')]].min().min(),
            max_value=df[[('1st type', 'score_1'), ('1st type', 'score_1')]].max().max(), format='%.4f'
        ),
        'score_2': st.column_config.ProgressColumn(
            'Test score',
            min_value=df[[('2nd type', 'score_2'), ('2nd type', 'score_2')]].min().min(),
            max_value=df[[('2nd type', 'score_2'), ('2nd type', 'score_2')]].max().max(), format='%.4f'
        )
    }
    st.dataframe(
        df_style,
        column_config=config,
        height=(df.shape[0] + 2) * 35 + 2,
        use_container_width=False
    )

st.markdown(
    '''
    ##  Model fine tuning

    Model fine-tuning is the final stage of model training. During this stage, both the model on top and base models 
    were trained simultaneously. The models face images datasets were used to do it.
    Otherwise, the process of performing this stage is similar to the 
    [previous](#model-on-top-learning) one.
    '''
)

with (st.expander('See details...')):
    st.markdown(
        '''
        ### Training settings
        '''
    )
    df = pd.DataFrame(
        [
            ['Optimizer', 'Kind of the learning algorithm', 'Adam', 'Adam'],
            ['Flip', 'Type of the random flipping of the input image', 'Horizontal', 'Horizontal'],
            ['Rotation', 'Random rotation range of the input image', '-0.1..0.1', '-0.1..0.1'],
            ['Zoom', 'Random resize range of the input image', '-0.2..0.2', '-0.2..0.2'],
            ['Contrast', 'Range of the random change of the contrast of the input image', '-0.2..0.2', '-0.2..0.2'],
            ['Brightness', 'Range of the random change of the brightness of the input image', '-0.2..0.2', '-0.2..0.2'],
            ['Initial Rate', 'Initial value of learning rate of the optimizer', 1e-4, 1e-4],
            ['Rate decay', 'Decay rate of learning rate of the optimizer', 0.96, 0.96],
            ['Loss', 'Objective that the model will try to minimize', 'Sparse Categorical Crossentropy',
             'Mean Absolute Error'],
            ['Metric', 'Metric for model''s prediction quality estimation', 'Sparse Categorical Accuracy',
             'Mean Absolute Percentage Error'],
            ['Epochs', 'Number of epochs to train the model', 50, 50],
            ['Patience', 'Maximum number of epochs without metric improvement', 3, 3],
        ],
        columns=['Parameter', 'Description', '1st type', '2nd type']
    ).set_index('Parameter')

    st.dataframe(
        df,
        height=(df.shape[0] + 1) * 35 + 2,
        use_container_width=False
    )

    st.markdown(
        '''
        ### Training charts
        '''
    )

    rows = []

    st.markdown(
        '''
        #### 1st type 
        '''
    )

    rows.append(st.empty())

    st.markdown(
        '''
        #### 2nd type 
        '''
    )

    rows.append(st.empty())

    losses = ['Sparse Categorical Crossentropy', 'Mean Absolute Error']
    metrics = ['Sparse Categorical Accuracy', 'Mean Absolute Error Percentage']

    for index, (row, loss, metric) in enumerate(zip(rows, losses, metrics)):
        with row:
            df = st.session_state.model_fine_tuning_logs.loc[:, index + 1]
            df.rename(
                columns={
                    'learning_rate': 'Learning Rate',
                    'loss': loss,
                    'metric': metric,
                },
                inplace=True
            )
            df.index.name = 'Epoch'
            df.columns.name = 'Tag'
            stacked_df = df.stack(future_stack=True).reset_index().rename(columns={0: 'Value'})
            stacked_df.info()

            fig = px.line(
                stacked_df, x='Epoch', y='Value',
                facet_col='Tag', facet_col_wrap=3,
                facet_col_spacing=0.075, facet_row_spacing=0.125,
                # height=600, width=830
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_yaxes(matches=None, visible=True, showticklabels=True)
            fig.update_layout(margin_t=60, margin_b=60)
            # fig.add_vline(x=df['Test Score'].idxmax(), line_dash='dash',
            #               line_color=DEFAULT_COLORS[0], label_text='Best Epoch')
            st.plotly_chart(fig, use_container_width=False)

    st.markdown('### Test scores')
    df = st.session_state.model_fine_tuning
    columns = pd.MultiIndex.from_tuples(
        [
            ('1st type', 'best_epoch'),
            ('1st type', 'loss'),
            ('1st type', 'metric'),
            ('1st type', 'score'),
            ('2nd type', 'best_epoch'),
            ('2nd type', 'loss'),
            ('2nd type', 'metric'),
            ('2nd type', 'score')
        ]
    )
    df.columns = columns
    df_style = df.style
    # df_style.highlight_max(
    #     subset=[('1st type', 'score'), ('2nd type', 'score')],
    #     props=f'color: white; background-color: {DEFAULT_COLORS[1]};'
    # )
    config = {
        'model_on_top_config': 'Model',
        'best_epoch': 'Best epoch',
        'loss': 'Loss',
        'metric': 'Metric',
        'score': st.column_config.ProgressColumn(
            'Test score',
            min_value=0, max_value=1, format='%.4f'
        ),
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
    
    In result of the work were built, trained and tested the following models:
    '''
)

df = pd.DataFrame(
    [
        ['EfficientNetB0',
         '1 dense layer with 1024 units', 0.5146],
        ['EfficientNetB2',
         '2 dense layers with 1024 and 512 units', 0.4846],
    ],
    index=pd.Index([1, 2], name='Type'),
    columns=['Base model', 'Model on top', 'Test score']
)

config = {
    # 'Base model': st.column_config.LinkColumn()
}

st.dataframe(
    df,
    column_config=config,
    use_container_width=False
)


