import re

import pandas as pd
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import plotly.graph_objects as go
from auxiliary import DEFAULT_COLORS


# Creating face detector
@st.cache_resource
def create_face_detector(haar_file_name = st.secrets['face-detector']['haar_file_name']):
    haar_file_path = (Path(cv2.__file__).parent / 'data' / haar_file_name).as_posix()
    face_detector = cv2.CascadeClassifier(haar_file_path)
    return face_detector


@st.cache_resource(show_spinner='Extracting face...')
def extract_face(
    image, _detector,
    scale_factor=st.secrets['face-detector']['scale_factor'],
    min_neighbors=st.secrets['face-detector']['min_neighbors'],
    min_face_size=st.secrets['face-detector']['min_face_size']
):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = _detector.detectMultiScale(gray_frame, scale_factor, min_neighbors, 0,
                                           (min_face_size, min_face_size))
    if len(face_rects) == 0:
        return
    x = y = w = h = 0
    for face_rect in face_rects:
        if face_rect[2] > w and face_rect[3] > h:
            x, y, w, h = face_rect
    face = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
    return np.asarray(face).tolist()


@st.cache_resource(show_spinner='Recognizing...')
def predict_probs(face):
    probs = st.session_state.google_cloud.endpoints[0].predict(
        instances=[face], use_dedicated_endpoint=True).predictions[0]
    return probs


@st.cache_resource(show_spinner='Recognizing...')
def predict_valence_arousal(face):
    valence, arousal = st.session_state.google_cloud.endpoints[1].predict(
        instances=[face], use_dedicated_endpoint=True).predictions[0]
    return valence, arousal


def plot_wheel_of_emotions(emotions=st.session_state.emotions, emotion=''):
    fig = go.Figure()
    valence_list = [v for v, _ in emotions.values()]
    arousal_list = [a for _, a in emotions.values()]
    marker_colors = [DEFAULT_COLORS[1] if e == emotion else DEFAULT_COLORS[0] for e in emotions]
    fig.add_scatter(
        x=valence_list, y=arousal_list,
        name='', mode='markers+text',
        text=list(emotions), textposition='top center',
        hovertemplate='valence: %{x:.2f}<br>arousal: %{y:.2f}',
        marker_color=marker_colors,
        showlegend=False
    )
    fig.add_shape(
        type='circle', x0=-1., y0=-1., x1=1., y1=1.,
        line_width=1, line_color='lightgray'
    )
    fig.update_xaxes(range=(-1.1, 1.1), title='valence', showgrid=True)
    fig.update_yaxes(range=(-1.1, 1.1), title='arousal')
    fig.update_layout(
        width=550,
        height=550,
        margin_t=25, margin_r=0, margin_b=0, margin_l=0
    )
    return fig


def main():
    st.markdown(
        '''
        # Camera
        '''
    )
    st.markdown('## Face picture')
    c1, _ = st.columns(2, gap='large')
    with c1:
        picture = st.camera_input('', label_visibility='collapsed')
        if picture is None:
            st.info('Please take a face photo.', icon=":material/info:")
            return
    c1, _ = st.columns(2, gap='large')
    with c1:
        st.markdown('## Emotion recognition')
        bytes_data = picture.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        detector = create_face_detector()
        face = extract_face(image, detector)
        if face is None:
            st.error('No face detected. Please take another photo.', icon=":material/error:")
            return
    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.markdown('### 1st type')
        df_emotions = pd.DataFrame()
        df_emotions['emotion'] = list(st.session_state.emotions)
        df_emotions.set_index('emotion', inplace=True)
        df_emotions['prob'] = predict_probs(face)
        df_emotions['color'] = DEFAULT_COLORS[0]
        df_emotions.at[df_emotions['prob'].idxmax(), 'color'] = DEFAULT_COLORS[1]
        fig = go.Figure(
            [
                go.Bar(
                    x=df_emotions.index, y=df_emotions['prob'],
                    texttemplate='%{y:.2f}',
                    marker={'color': df_emotions['color']}
                )
            ],
        )
        fig.update_layout(
            height=352, width=502,
            margin_t=25, margin_r=0, margin_b=0, margin_l=0
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('### 2nd type')
        valence, arousal = predict_valence_arousal(face)
        distances = [
            ((valence - v) ** 2 + (arousal - a) ** 2) ** 0.5
            for v, a in st.session_state.emotions.values()
        ]
        d = min(distances)
        e = list(st.session_state.emotions)[distances.index(d)]
        fig = plot_wheel_of_emotions(emotion=e)
        fig.add_scatter(
            x=[valence], y=[arousal],
            name='',
            mode='markers',
            hovertemplate='valence: %{x:.2f}<br>arousal: %{y:.2f}',
            marker_color=DEFAULT_COLORS[1],
            showlegend=False
        )
        fig.add_shape(
            type='circle', x0=valence - d, y0=arousal - d, x1=valence + d, y1=arousal + d,
            line_width=1, line_color='gray', line_dash='dot'
        )
        st.plotly_chart(fig, use_container_width=False)


main()