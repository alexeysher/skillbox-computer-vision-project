from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from auxiliary import DEFAULT_COLORS
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



st.markdown(
    '''
    # Image preprocessing
    
    This pipeline is designed to prepare training and test datasets. 
    Additionally, images of faces that are too similar to each other and images of faces whose labeling 
    is questionable are excluded from the training dataset.

    '''
)

st.markdown(
    '''
    ## Extracting face images from the train dataset
    
    This stage has a goal to remove unnecessary details from the images that can "confuse" the model, 
    leaving only the face areas.
    
    During the stage execution, train images were cropped to the face area. 
    Faces in the images were searched using the pretrained face detector based on 
    [Cascade Classifier](https://docs.opencv.org/4.11.0/db/d28/tutorial_cascade_classifier.html) 
    provided with [OpenCV](http://opencv.org) library.

    Images on witch no face was detected were left in their original state.
    '''
)

with st.expander('See details...'):

    st.markdown('### Settings of Face Detector')

    df = pd.DataFrame(
        [
            [
                'Scale factor',
                'Specify how much the image size is reduced at each image scale',
                1.1,
            ],
            [
                'Min. neighbors',
                'Specify how many neighbors each candidate rectangle should have to retain it',
                3,
            ],
            [
                'Min. size',
                'Minimum possible object size. Objects smaller than that are ignored.',
                'A half of the biggest dimension of the image',
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
        ### Examples of extracted faces  
        '''
    )

    df = st.session_state.train_faces[[(1, 'extracted_url')]].transpose()
    config = {
        emotion: st.column_config.ImageColumn(emotion)
        for emotion in st.session_state.emotions
    }
    st.dataframe(
        df,
        hide_index=True,
        # column_config=config,
        use_container_width=True,
        row_height=100
    )

    st.markdown(
        '''
        ### Examples of images on witch no face was detected
        '''
    )

    df = st.session_state.train_faces[[(1, 'not_extracted_url')]].transpose()
    config = {
        emotion: st.column_config.ImageColumn(emotion)
        for emotion in st.session_state.emotions
    }
    st.dataframe(
        df,
        hide_index=True,
        column_config=config,
        use_container_width=True,
        row_height=100
    )

st.markdown(
    '''
    '''
)

st.markdown(
    '''
    ## Extracting face images from the test dataset
    
    Similar to [previous](#extracting-face-images-from-the-train-dataset) stage during the execution of the stage
    the test dataset of face images was build.
    
    Within this work all images were successfully processed.  
    '''
)

st.markdown(
    '''
    ## Feature extraction from the train dataset

    The goal of this step is to obtain a dataset from the feature vectors of the training dataset's facial images. 
    This dataset is used in 
    [the selecting the best model on top](model_building#selecting-the-best-model-on-top) stage of 
    the [model building](model_building#model-building) pipeline. 
    Features were extracted by "passing" face images through the base model.
    '''
)

with st.expander('See details...'):
    st.markdown(
        '''
        For each class the set of feature vectors was gotten:
        '''
    )
    st.latex(
        r'''
        V = \{v_i\},
        \text{where $v_i$ - feature vector of $i$th image of the class}
        '''
    )

st.markdown(
    '''
    ## Feature extraction from the test dataset
    
    The goal of this step is to obtain a dataset from the feature vectors of the test dataset faces. 
    This dataset is also used in the step of selecting the best fully connected model. 
    This step is performed similarly to the [previous](#feature-extraction-from-the-train-dataset) 
    one and has the same set of tuning parameters.
    '''
)

st.markdown(
    '''
    ## Cleaning up the training dataset

    The goal of this stage is to exclude from the training dataset of face images objects 
    that may negatively affect the quality of model training:
    - identical or very similar face images;
    - face images that may have been incorrectly labeled.
    
    For measuring similarity of images purpose the [cosine distance](https://wiki5.ru/wiki/Cosine_similarity)
    was used.
    '''
)
with st.expander('See details...'):
    st.markdown(
        '''
        For cleaning purpose the set of the similarity of each pair of the feature vectors 
        within each class (emotion) were found: 
        '''
    )
    st.latex(
        r'''
        S = \{s_{i,j}\},
        '''
    )
    st.markdown(
        '''
        where $s_{i,j}$ - similarity of the $i$th and the $j$th images of the class: 
        '''
    )
    st.latex(
        r'''
        s_{i,j} = \left\{\dfrac{v_i \cdot v_j}{\|v_i\| \cdot \|v_j\|}\right\},
        '''
    )

    st.markdown(
        r'''
        where $v_i$ and $v_j$ are the feature vectors the $i$th and the $j$th images.
        '''
    )

st.markdown(
    '''
    Too similar face images were identified according to the following rule. Such images include pairs of images, 
    the [cosine distance](https://wiki5.ru/wiki/Cosine_similarity) between which exceeds the upper reliability limit.
    Only one image with the higher index from each such pair were excluded from the train datasets.
    '''
)

with st.expander('See details...'):

    st.markdown(
        '''
        This upper reliability limit ${s}_{max}$ is calculated for each emotion independently. The limit is
        3 times standard deviation $\sigma$ above the mean value $\overline{s}$:
        '''
    )
    st.latex(
        r'''
        s_{max}=\overline{s} + 3\sigma.
        '''
    )
    st.markdown(
        '''
        The $i$th image marked as duplicated and was excluded from datasets if following condition was truth for it:
        '''
    )
    st.latex(
        r'''
        \text{any }s_{i,j} > s_{max},\text{ where }j > i.
        '''
    )

    st.markdown('### Distribution of similarity')

    col_1, col_2 = st.columns(2, gap='large')
    with col_1:
        st.markdown('#### 1st type')
    with col_2:
        st.markdown('#### 2nd type')
    for index, col in enumerate((col_1, col_2)):
        with col:
            fig = make_subplots(3, 3, subplot_titles=list(st.session_state.emotions))
            for plot_index, (emotion, similarity_max, (similarity_counts, similarity_bins)) in enumerate(
                    zip(
                        st.session_state.emotions,
                        st.session_state.similarity_limits[index],
                        st.session_state.similarity_hists[index]
                    )
            ):
                similarity_bins = 0.5 * (similarity_bins[:-1] + similarity_bins[1:])
                hovers = [
                    f'Similarity: {x0:.3f} - {x1:.3f}<br>Count: {y:d}' for x0, x1, y in zip(
                        similarity_bins[:-1], similarity_bins[1:], similarity_counts
                    )
                ]
                row = plot_index // 3 + 1
                col = plot_index % 3 + 1
                # print(f'{plot_index=}, {emotion=}, {row=}, {col=}')
                fig = fig.add_trace(
                    go.Bar(
                        x=similarity_bins, y=similarity_counts, name=emotion, marker_color=DEFAULT_COLORS[plot_index]
                    ),
                    row=row, col=col
                )
                # fig.update_xaxes(title=None)
                # fig.update_yaxes(title=None)
                fig.update_traces(hovertext=hovers, hoverinfo='text', hovertemplate=None)
                fig.add_vline(
                    x=similarity_max, row=row, col=col,
                    line_color=DEFAULT_COLORS[1], line_dash='dash', line_width=1,
                    label_text=f's<sub>max</sub>={similarity_max:.3f}'
                )
                fig.update_layout(margin_t=60, margin_b=0)
            st.plotly_chart(fig, key=f'similarity_{index}_{emotion}')

    col_1, col_2 = st.columns(2, gap='large')
    with col_1:
        st.markdown('#### 1st type')
    with col_2:
        st.markdown('#### 2nd type')
    df = st.session_state.train_face_cleaning
    for index, col in enumerate((col_1, col_2)):
        with col:
            fig = go.Figure()
            # fig.add_bar(name='Processed', x=df.index, y=df['faces_num'], marker_color=DEFAULT_COLORS[0])
            fig.add_bar(name='', x=df.index, y=df[(index + 1, 'duplicated_percent')],
                        marker_color=DEFAULT_COLORS[index],
                        text=df[(index + 1, 'duplicated_number')],
                        textfont_color='white')
            fig.update_xaxes(title=None)
            fig.update_yaxes(title='Percentage of duplicated faces')
            fig.update_traces(hovertemplate='emotion: <b>%{x}</b><br>percentage: <b>%{y:.2f}</b><br>count: <b>%{text}</b>')
            fig.update_layout(font_size=12, margin_t=20, margin_b=0, width=600, height=(df.shape[0] + 1) * 35 + 2)
            st.plotly_chart(fig, use_container_width=False)

    st.markdown(
        '''
        ### Examples of too similar face images  
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

    for index, row in enumerate(rows):
        with row:
            df = st.session_state.train_duplicated_faces[index+1].transpose()
            config = {
                emotion: st.column_config.ImageColumn(emotion)
                for emotion in st.session_state.emotions
            }
            st.dataframe(
                df,
                hide_index=True,
                column_config=config,
                use_container_width=True,
                row_height=100
            )

st.markdown(
    '''
    To identify too different face images, 
    the median value of [cosine distances](https://en.wikipedia.org/wiki/Cosine_similarity) between the feature vector 
    of this image and the feature vectors of the other face images with the same label 
    is additionally calculated for each image. 
    Images that are too different are those whose median similarity value 
    with other images falls outside the lower upper reliability limit.
    '''
)

with st.expander('See details...'):
    st.latex(
        r'''
        \tilde{s}_{i}=med(\{s_{i,j}\})
        '''
    )
    st.markdown(
        '''
        This upper reliability limit ${s}_{max}$ is calculated for each emotion independently. The limit is
        3 times standard deviation $\sigma$ above the mean value $\overline{s}$:
        '''
    )
    st.latex(
        r'''
        \tilde{s}_{min}=\overline{\tilde{s}} - 3\sigma.
        '''
    )

    st.markdown(
        '''
        The $i$th image marked as different and was excluded from datasets if following condition was truth for it:
        '''
    )
    st.latex(
        r'''
        \tilde{s_i} < \tilde{s_{min}}.
        '''
    )

    st.markdown('### Distribution of medians of similarity')

    for index, col in enumerate((col_1, col_2)):
        with col:
            fig = make_subplots(3, 3, subplot_titles=list(st.session_state.emotions))
            # for plot_index, (emotion, similarity_max, (similarity_counts, similarity_bins)) in enumerate(
            #         zip(
            #             st.session_state.emotions,
            #             st.session_state.similarity_limits[index],
            #             st.session_state.similarity_hists[index]
            #         )
            # ):
            #     similarity_bins = 0.5 * (similarity_bins[:-1] + similarity_bins[1:])
            #     hovers = [
            #         f'Similarity: {x0:.3f} - {x1:.3f}<br>Count: {y:d}' for x0, x1, y in zip(
            #             similarity_bins[:-1], similarity_bins[1:], similarity_counts
            #         )
            #     ]
            #     row = plot_index // 3 + 1
            #     col = plot_index % 3 + 1
            #     print(f'{plot_index=}, {emotion=}, {row=}, {col=}')
            #     fig = fig.add_trace(
            #         go.Bar(
            #             x=similarity_bins, y=similarity_counts, name=emotion, marker_color=DEFAULT_COLORS[plot_index]
            #         ),
            #         row=row, col=col
            #     )
            #     # fig.update_xaxes(title=None)
            #     # fig.update_yaxes(title=None)
            #     fig.update_traces(hovertext=hovers, hoverinfo='text', hovertemplate=None)
            #     fig.add_vline(
            #         x=similarity_max, row=row, col=col,
            #         line_color=DEFAULT_COLORS[1], line_dash='dash', line_width=1,
            #         label_text=f's<sub>max</sub>={similarity_max:.3f}'
            #     )
            #     fig.update_layout(margin_t=60, margin_b=0)
            # st.plotly_chart(fig, key=f'similarity_{index}_{emotion}')

    col_1, col_2 = st.columns(2, gap='large')
    with col_1:
        st.markdown('#### 1st type')
    with col_2:
        st.markdown('#### 2nd type')
    for index, col in enumerate((col_1, col_2)):
        with col:
            fig = make_subplots(3, 3, subplot_titles=list(st.session_state.emotions))
            for plot_index, (emotion, similarity_median_min, (similarity_medians_counts, similarity_medians_bins)) in enumerate(
                    zip(
                        st.session_state.emotions,
                        st.session_state.similarity_medians_limits[index],
                        st.session_state.similarity_medians_hists[index]
                    )
            ):
                similarity_medians_bins = 0.5 * (similarity_medians_bins[:-1] + similarity_medians_bins[1:])
                hovers = [
                    f'Similarity: {x0:.3f} - {x1:.3f}<br>Count: {y:d}' for x0, x1, y in zip(
                        similarity_medians_bins[:-1], similarity_medians_bins[1:], similarity_medians_counts
                    )
                ]
                row = plot_index // 3 + 1
                col = plot_index % 3 + 1
                # print(f'{plot_index=}, {emotion=}, {row=}, {col=}')
                fig = fig.add_trace(
                    go.Bar(
                        x=similarity_medians_bins, y=similarity_medians_counts,
                        name=emotion, marker_color=DEFAULT_COLORS[plot_index]
                    ),
                    row=row, col=col
                )
                # fig.update_xaxes(title=None)
                # fig.update_yaxes(title=None)
                fig.update_traces(hovertext=hovers, hoverinfo='text', hovertemplate=None)
                fig.add_vline(
                    x=similarity_median_min, row=row, col=col,
                    line_color=DEFAULT_COLORS[1], line_dash='dash', line_width=1,
                    label_text=f's&#x342;<sub>min</sub>={similarity_median_min:.3f}'
                )
                fig.update_layout(margin_t=60, margin_b=0)
            st.plotly_chart(fig, key=f'similarity_medians_{index}_{emotion}')

    st.markdown('### Percentage of too different images')
    df = st.session_state.train_face_cleaning

    col_1, col_2 = st.columns(2, gap='large')
    with col_1:
        st.markdown('#### 1st type')
    with col_2:
        st.markdown('#### 2nd type')

    for index, col in enumerate((col_1, col_2)):
        with col:
            fig = go.Figure()
            # fig.add_bar(name='Processed', x=df.index, y=df['faces_num'], marker_color=DEFAULT_COLORS[0])
            fig.add_bar(name='', x=df.index, y=df[(index + 1, 'different_percent')],
                        marker_color=DEFAULT_COLORS[index],
                        text=df[(index + 1, 'different_number')],
                        textfont_color='white')
            fig.update_xaxes(title=None)
            fig.update_traces(hovertemplate='emotion: <b>%{x}</b><br>percentage: <b>%{y:.2f}</b><br>count: <b>%{text}</b>')
            fig.update_yaxes(title='Percentage of different faces')
            fig.update_layout(font_size=12, margin_t=20, margin_b=0, width=600, height=(df.shape[0] + 1) * 35 + 2)
            st.plotly_chart(fig, use_container_width=False, key=f'similarity_medians_percent_{index}_{emotion}')

    st.markdown(
        '''
        ### Examples of too different face images
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

    for index, row in enumerate(rows):
        with row:
            df = st.session_state.train_different_faces[index+1].transpose()
            config = {
                emotion: st.column_config.ImageColumn(emotion)
                for emotion in st.session_state.emotions
            }
            st.dataframe(
                df,
                hide_index=True,
                column_config=config,
                use_container_width=True,
                row_height=100
            )

st.markdown(
    '''
    ## Results
    '''
)

df = st.session_state.train_face_cleaning

col_1, col_2 = st.columns(2, gap='large')
with col_1:
    st.markdown('#### 1st type')
with col_2:
    st.markdown('#### 2nd type')

for index, col in enumerate((col_1, col_2)):
    with col:
        fig = go.Figure()
        fig.add_bar(name='Remain', x=df.index, y=df[(index + 1, 'remain_number')],
                    marker_color=DEFAULT_COLORS[0],
                    text=df[(index + 1, 'remain_percent')],
                    texttemplate='%{text:.1f}%')
        fig.add_bar(name='Failed', x=df.index, y=df[(index + 1, 'failed_number')],
                    marker_color=DEFAULT_COLORS[1],
                    text=df[(index + 1, 'failed_percent')],
                    texttemplate='%{text:.1f}%')
        fig.update_traces(hovertemplate='emotion: <b>%{x}</b><br>count: <b>%{y}</b><br>percentage: <b>%{text:.1f}</b>')
        fig.update_xaxes(title='Emotion')
        fig.update_yaxes(title='Number of faces')
        fig.update_layout(barmode='stack', margin_t=60, margin_b=0, height=400)
        st.plotly_chart(fig, key=f'face_cleaning_{index}')

