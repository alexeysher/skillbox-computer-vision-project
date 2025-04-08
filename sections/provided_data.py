import pandas as pd
import streamlit as st
from auxiliary import DEFAULT_COLORS
import plotly.graph_objects as go

st.markdown(
    '''
    # Provided Data
    
    ### Training data
    
    The dataset attached to the 
    [skillbox-computer-vision-project](https://www.kaggle.com/competitions/skillbox-computer-vision-project/data) 
    competition was used to train the model. The dataset is a set of files distributed across folders 
    with emotion names:
    '''
)


df = st.session_state.train_size.reset_index()
df['color'] = DEFAULT_COLORS[0]
# df.at[df['percent'].idxmax(), 'color'] = DEFAULT_COLORS[2]
# df.at[df['percent'].idxmin(), 'color'] = DEFAULT_COLORS[1]
fig = go.Figure()
fig.add_bar(x=df['emotion'], y=df['percent'],
            marker_color=df['color'],
            text=df['count'], textfont_color='white', name='')
fig.update_xaxes(title=None)
fig.update_yaxes(title='Percentage')
fig.update_traces(hovertemplate='emotion: <b>%{x}</b><br>percentage: <b>%{y:.2f}</b><br>count: <b>%{text}</b>')
fig.update_layout(margin_t=40, margin_b=0, width=600, height=(df.shape[0] + 1) * 35 + 2)
st.plotly_chart(fig, use_container_width=False)



st.markdown(
    '''
    Such dataset is suitable only for training the 1st type model.
    To train the 2nd type model images were labeled with typical values of the valence and arousal of
    emotion witch they belongs:
    '''
)

col1, col2 = st.columns([17, 83])

with col1:
    df = pd.DataFrame(columns=['emotion', 'valence', 'arousal'])
    df[['emotion', 'valence', 'arousal']] = [
        (emotion, valence, arousal) for emotion, (valence, arousal) in st.session_state.emotions.items()
    ]
    config = {
        'emotion': 'Emotion',
        'valence': 'Valence',
        'arousal': 'Arousal',
    }
    df.set_index('emotion', inplace=True)
    st.dataframe(df, column_config=config, use_container_width=False)

with col2:
    fig = go.Figure()
    fig.add_scatter(
        x=df['valence'], y=df['arousal'],
        name='', mode='markers+text',
        text=df.index, textposition='top center',
        hovertemplate='valence: %{x:.2f}<br>arousal: %{y:.2f}',
        showlegend=False
    )
    fig.add_shape(
        type='circle', x0=-1., y0=-1., x1=1., y1=1.,
        line_width=1, line_color='lightgray'
    )
    fig.update_xaxes(range=(-1.1, 1.1), title='Valence', showgrid=True)
    fig.update_yaxes(range=(-1.1, 1.1), title='Arousal')
    fig.update_layout(
        width=(df.shape[0] + 1) * 35,
        height=(df.shape[0] + 1) * 35,
        margin_t=25, margin_r=0, margin_b=0, margin_l=0
    )
    st.plotly_chart(fig, use_container_width=False)

st.markdown(
    '''
    ### Test data
    
    To evaluate the quality of the model during the training process, the dataset attached to the competition 
    [skillbox-computer-vision-project](https://www.kaggle.com/competitions/skillbox-computer-vision-project/data) 
    is also used. The dataset is also a set of files. It contains 5000 items.But they have no label. 
    
    The quality of the model's prediction can only be assessed on the platform itself.
    The evaluation perform by categorisation accuracy metric:    
    '''
)

st.latex(
    r'''
    \dfrac{\sum{[y_{true} == y_{pred}]}}{len(y_{true})}
    '''
)

st.markdown(
    '''
    Thus a test submission should contains labeled by emotion for each test image.
    
    So in the test submission for the 1st type model each image was labeled by the most probable emotion from prediction.
    In the test submission for the 2nd type model each image was labeled by emotion
    with nearest combination of typical values of valence and arousal to predicted ones. 
    '''
)
