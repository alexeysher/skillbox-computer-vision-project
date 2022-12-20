from typing import Union

from pathlib import Path
from datetime import timedelta, datetime, date
from collections import namedtuple

import pandas as pd
import streamlit as st
import cv2
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go

from emotion_recognition import FaceEmotionRecognitionNet, EMOTIONS
from fourcc import FourCC

MODEL_PATH = 'D:/Aff-wild2/model/'
ANNOTATIONS_PATH = 'D:/Aff-wild2/annotations/'
VIDEO_PATH = 'D:/Aff-wild2/video/'
TRAILERS_PATH = 'D:/Aff-wild2/trailers/'
HAAR_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 3
MIN_FACE_SIZE = 64


def hide_menu_button():
    st.markdown(
        """
        <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )


def remove_blank_space():
    # return
    st.markdown(f'''
                <style>
                    .css-18e3th9 {{
                        padding-top: 1.0rem;
                    }}
                </style>
                <style>
                    .css-1vq4p4l {{
                        padding-top: 4.0rem;
                    }}
                </style>
                ''', unsafe_allow_html=True,
                )


VideoInfo = namedtuple(
    'VideoInfo',
    ['frame_width', 'frame_height',
     'frame_rate', 'frames_number',
     'fourcc', 'fourcc_str',
     'duration', 'duration_str']
)

VideoData = namedtuple('VideoData', ['bytes', 'file_name'])

HyperParams = namedtuple('HyperParams',
                         ['tma_window', 'min_arousal', 'min_prominence', 'min_duration', 'rel_height', 'min_distance'])


@st.experimental_singleton(show_spinner='Создание детектора лиц...')
def create_face_detector():
    detector = cv2.CascadeClassifier(HAAR_FILE)
    return detector


@st.experimental_singleton(show_spinner='Загрузка модели распознавания эмоций...')
def create_emotion_recognizer() -> FaceEmotionRecognitionNet:
    return FaceEmotionRecognitionNet(MODEL_PATH, EMOTIONS)


def upload_video() -> VideoData | None:
    """Загрузка видео. Загруженный файл сохраняется в папку 'temp' и отображается в проигрывателе.
    Возвращает путь к загруженному файлу в виде объекта Path."""
    with st.expander('Загрузка видео...'):
        uploaded_file = st.file_uploader('Загрузка видео', ['mp4', 'avi'], label_visibility='hidden')
    if uploaded_file is None:
        return None
    video_bytes = uploaded_file.read()
    return VideoData(video_bytes, uploaded_file.name)


def save_video(video_data: VideoData) -> str:
    video_dir = Path(VIDEO_PATH)
    if not video_dir.exists():
        video_dir.mkdir()
    video_file = video_dir / video_data.file_name
    if not video_file.exists():
        with open(video_file, mode='wb') as video:
            video.write(video_data.bytes)
    return video_file.as_posix()


def create_video_capture(file_path: str) -> cv2.VideoCapture:
    """Создание "захватчика" видео файла."""
    capture = cv2.VideoCapture(file_path)
    if capture is None:
        return
    if not capture.isOpened():
        return
    return capture


def retrieve_video_info(capture: cv2.VideoCapture) -> Union[VideoInfo, None]:
    """Возвращает информацию о видео открытом в 'захватчике'."""
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
    frames_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = FourCC[fourcc.to_bytes(4, 'little').decode('ascii')]
    duration = timedelta(seconds=(frames_number - 1) / frame_rate)
    duration_str = f'{duration.seconds // 60:02d}:{duration.seconds % 60:02d}'
    info = VideoInfo(frame_width, frame_height,
                     frame_rate, frames_number,
                     fourcc, fourcc_str,
                     duration, duration_str)
    return info


def extract_face(image, detector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rect = detector.detectMultiScale(gray_image, SCALE_FACTOR, MIN_NEIGHBORS, 0, (MIN_FACE_SIZE, MIN_FACE_SIZE))
    if len(faces_rect) == 0:
        return
    x, y, w, h = faces_rect[0]
    face_image = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
    return face_image


def recognize_video_emotions(
        file_path: str,
        video_info: VideoInfo,
        video_capture: cv2.VideoCapture,
        face_detector,
        emotion_recognizer: FaceEmotionRecognitionNet) -> [float]:
    annotations_path = Path(ANNOTATIONS_PATH)
    if not annotations_path.exists():
        annotations_path.mkdir()
    annotation_file = annotations_path / Path(file_path).with_suffix('.csv').name
    if annotation_file.exists():
        arousals = pd.read_csv(annotation_file)['arousal'].to_list()
        return arousals
    frame_index = 0
    faces_number = 0
    arousal = 0.0
    arousals = []
    empty_1 = st.empty()
    empty_1.markdown('Распознавание эмоций...')
    empty_2 = st.empty()
    progress_bar = empty_2.progress(0.0)
    while True:
        ret, image = video_capture.read()
        if not ret:
            break
        face_image = extract_face(image, face_detector)
        if face_image is not None:
            faces_number += 1
            arousal = emotion_recognizer.predict(face_image)[-1]
        arousals.append(arousal)
        frame_index += 1
        progress_bar.progress(frame_index / video_info.frames_number)
    video_capture.release()
    empty_1.empty()
    empty_2.empty()
    pd.DataFrame(arousals, columns=['arousal']).to_csv(annotation_file)
    return arousals


def set_hyperparams(video_info: VideoInfo) -> HyperParams:
    # st.markdown('Параметры выбора фрагментов')
    tma_window = int(st.slider('Размер окна TMA [с]', 0.05, 2.0, 1.0, 0.05) *
                     video_info.frame_rate)
    min_arousal = st.slider('Мин. значение пика', 0.0, 1.0, 0.5, 0.05)
    min_prominence = st.slider('Мин. подъем', 0.0, 0.5, 0.25, 0.05)
    min_duration = int(st.slider('Мин. длительность [c]', 0.0, 1.0, 2.0, 0.05) *
                       video_info.frame_rate)
    rel_height = st.slider('Отн. высота границ', 0.0, 1.0, 0.5, 0.05)
    min_distance = int(st.slider('Мин. дистанция [c]', 0, 20, 10, 1) *
                       video_info.frame_rate)
    hyperparams = HyperParams(tma_window, min_arousal, min_prominence, min_duration, rel_height, min_distance)
    return hyperparams


def find_fragments(video_info: VideoInfo, arousals: [], hyperparams: HyperParams):
    today = date.today()
    start_date = datetime(today.year, today.month, today.day)
    end_date = start_date + video_info.duration
    trend = pd.DataFrame()
    trend['time'] = pd.date_range(start_date, end_date, periods=video_info.frames_number)
    trend['arousal'] = arousals
    trend['arousal_sma'] = trend['arousal'] \
        .rolling(window=hyperparams.tma_window, min_periods=1, center=True).mean()
    trend['arousal_tma'] = trend['arousal_sma'] \
        .rolling(window=hyperparams.tma_window, min_periods=1, center=True).mean()
    peaks, properties = find_peaks(
        trend['arousal_tma'],
        height=hyperparams.min_arousal,
        distance=hyperparams.min_distance,
        prominence=hyperparams.min_prominence,
        width=hyperparams.min_duration,
        rel_height=hyperparams.rel_height,
    )
    fragments = pd.DataFrame()
    fragments['peak_frame'] = peaks
    fragments['peak_time'] = trend.loc[fragments['peak_frame'].to_list(), 'time'].to_list()
    fragments['peak_arousal'] = properties['peak_heights']
    fragments['start_frame'] = [int(left_ips) for left_ips in properties['left_ips']]
    fragments['start_time'] = trend.loc[fragments['start_frame'].to_list(), 'time'].to_list()
    fragments['start_arousal'] = trend.loc[fragments['start_frame'].to_list(), 'arousal_tma'].to_list()
    fragments['end_frame'] = [int(left_ips) for left_ips in properties['right_ips']]
    fragments['end_time'] = trend.loc[fragments['end_frame'].to_list(), 'time'].to_list()
    fragments['end_arousal'] = trend.loc[fragments['end_frame'].to_list(), 'arousal_tma'].to_list()
    return trend, fragments


def plot_chart(trend: pd.DataFrame, fragments: pd.DataFrame):
    st.markdown('##### Динамика интенсивности эмоций')
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend['time'], y=trend['arousal'],
            mode='lines', name='Интенсивность',
            opacity=0.5,
            line={'width': 1, 'shape': 'spline', 'color': px.colors.qualitative.G10[1]}
        )
    )
    for fragment, (start_frame, end_frame, peak_time, start_time, end_time) in fragments[
        ['start_frame', 'end_frame', 'peak_time', 'start_time', 'end_time']
    ].iterrows():
        fig.add_trace(
            go.Scatter(
                x=trend.loc[start_frame: end_frame, 'time'],
                y=trend.loc[start_frame: end_frame, 'arousal_tma'],
                mode='lines', name=None,
                line={'width': 0, 'shape': 'spline', 'color': px.colors.qualitative.G10[7]},
                fill='toself',
                showlegend=False,
            )
        )
        # fig.add_vline(
        #     peak_time,
        #     line={'width': 1, 'dash': 'dash', 'color': px.colors.qualitative.G10[3]}
        # )
        # fig.add_vline(
        #     start_time,
        #     line={'width': 1, 'dash': 'dash', 'color': px.colors.qualitative.G10[3]}
        # )
        # fig.add_vline(
        #     end_time,
        #     line={'width': 1, 'dash': 'dash', 'color': px.colors.qualitative.G10[3]}
        # )
    fig.add_trace(
        go.Scatter(
            x=trend['time'], y=trend['arousal_tma'],
            mode='lines', name='TMA интенсивность',
            line={'width': 2, 'shape': 'spline', 'color': px.colors.qualitative.G10[3]}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fragments['start_time'], y=fragments['start_arousal'],
            mode='markers', name='Начало фрагмента',
            marker={
                'symbol': 'line-nw-open',
                'size': 6, 'color': px.colors.qualitative.G10[3],
                'line': {'width': 2, 'color': px.colors.qualitative.G10[3]}
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fragments['end_time'], y=fragments['end_arousal'],
            mode='markers', name='Конец фрагмента',
            marker={
                'symbol': 'line-ne-open',
                'size': 6, 'color': px.colors.qualitative.G10[3],
                'line': {'width': 2, 'color': px.colors.qualitative.G10[3]}
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fragments['peak_time'], y=fragments['peak_arousal'],
            mode='markers', name='Пик интенсивности',
            marker={
                'symbol': 'line-ns-open',
                'size': 6, 'color': px.colors.qualitative.G10[3],
                'line': {'width': 2, 'color': px.colors.qualitative.G10[3]}
            },
        )
    )
    fig.update_xaxes(tickformat="%M:%S")
    fig.update_layout(legend={'orientation': 'h', 'yanchor': 'top', 'y': 1.2, 'xanchor': 'right', 'x': 1.0})
    st.plotly_chart(fig, use_container_width=True)


def create_trailer_writer(video_file_path, video_info: VideoInfo) -> cv2.VideoWriter:
    """Создание "захватчика" видео файла."""
    trailers_dir = Path(TRAILERS_PATH)
    if not trailers_dir.exists():
        trailers_dir.mkdir()
    trailer_path = (trailers_dir / Path(video_file_path).name).as_posix()
    writer = cv2.VideoWriter(
        trailer_path, video_info.fourcc, video_info.frame_rate,
        (video_info.frame_width, video_info.frame_height))
    return writer, trailer_path


def display_fragments_table(fragments: pd.DataFrame):
    st.markdown('##### Отобранные фрагменты')
    df = fragments[['start_time', 'end_time']]
    df['duration'] = df['end_time'] - df['start_time']
    df['duration'] = df['duration'].dt.total_seconds().round(3).apply(str)
    df['start_time'] = df['start_time'].dt.strftime('%M:%S') + '.' + (df['start_time'].dt.microsecond // 1000).apply(str)
    df['end_time'] = df['end_time'].dt.strftime('%M:%S') + '.' + (df['end_time'].dt.microsecond // 1000).apply(str)
    df.rename(columns={'start_time': 'Начало', 'end_time': 'Конец', 'duration': 'Длительность [c]', }, inplace=True)
    df.index = range(1, df.shape[0]+1)
    df.index.name = 'Номер'
    st.table(df)


def save_trailer(video_capture, trailer_writer, fragments: pd.DataFrame):
    trailer_frames_number = (fragments['end_frame'] - fragments['start_frame'] + 1).sum()
    frames_number = 0
    empty = st.empty()
    progress_bar = empty.progress(0.0)
    for fragment, (start_frame, end_frame) in fragments[['start_frame', 'end_frame']].iterrows():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame + 1):
            _, frame = video_capture.read()
            trailer_writer.write(frame)
            frames_number += 1
            progress_bar.progress(frames_number / trailer_frames_number)
    empty.empty()
    video_capture.release()
    trailer_writer.release()


def download_trailer(trailer_path: str):
    path = Path(trailer_path)
    with open(path, mode='rb') as trailer_file:
        st.download_button('Скачать трейлер...', trailer_file, path.name, 'video' + path.suffix[1:])


def main():
    st.markdown('## Создание трейлеров')
    face_detector = create_face_detector()
    emotion_recognizer = create_emotion_recognizer()
    video_col_1, trailer_col_1 = st.columns(2, gap='large')
    with video_col_1:
        st.markdown('### Видео')
        video_data = upload_video()
    with trailer_col_1:
        st.markdown('### Трейлер')
        if video_data is None:
            st.info('Необходимо загрузить видео.')
            return
    video_path = save_video(video_data)
    video_capture = create_video_capture(video_path)
    video_col_2, trailer_col_2 = st.columns(2, gap='large')
    with video_col_2:
        st.video(video_data.bytes)
    video_info = retrieve_video_info(video_capture)
    video_col_3, trailer_col_3 = st.columns(2, gap='large')
    with video_col_3:
        arousals = recognize_video_emotions(video_path, video_info, video_capture, face_detector, emotion_recognizer)
    with st.sidebar:
        hyperparams = set_hyperparams(video_info)
    trend, fragments = find_fragments(video_info, arousals, hyperparams)
    with video_col_3:
        plot_chart(trend, fragments)
    with trailer_col_1:
        if fragments.empty:
            st.warning('Не найдено ни одного фрагмента.')
            return
    video_capture = create_video_capture(video_path)
    trailer_writer, trailer_path = create_trailer_writer(video_path, video_info)
    with trailer_col_2:
        save_trailer(video_capture, trailer_writer, fragments)
        st.video(trailer_path)
    with trailer_col_3:
        display_fragments_table(fragments)
    with trailer_col_1:
        download_trailer(trailer_path)


st.set_page_config(layout='wide')
hide_menu_button()
remove_blank_space()
main()
