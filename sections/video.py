import hashlib
import pickle
import math
from datetime import timedelta, datetime
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks, medfilt2d, butter, filtfilt
import subprocess
from auxiliary import GoogleCloud
from plotly.subplots import make_subplots
from auxiliary import DEFAULT_COLORS


class VideoData(bytes):
    """
    Video data.
    """


class VideoInfo:
    """
    Video info:
    frame_width [int] -
    frame_height [int] -
    frame_rate [float] -
    frames_number [int] -
    fourcc [float] -
    duration [timedelta] -
    duration_str [str] -
    """

    def __init__(self, file_name: str):
        capture = cv2.VideoCapture(file_name)
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
        self.frames_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        capture.release()
        self.duration = timedelta(seconds=self.frames_number / self.frame_rate)
        self.duration_str = f'{self.duration.seconds // 60:02d}:{self.duration.seconds % 60:02d}'

    def __str__(self):
        return (f'{self.frame_width=}, {self.frame_height=}, {self.frame_rate=}, {self.frames_number=}, '
                f'{self.fourcc=}, {self.duration=}, {self.duration_str=}')

    def __repr__(self):
        return self.__str__()


class Video:

    _temp_file_path = 'data/temp.mp4'

    def _compute_md5(self, algorithm='md5') -> str:
        """Computes the hash of video data using the specified algorithm."""
        hash_func = hashlib.new(algorithm)
        hash_func.update(self._data)
        return hash_func.hexdigest()

    def _save(self):
        with open(self.file_name, mode='wb') as video:
            video.write(self._data)

    def __init__(self, file_name: str | Path, data: VideoData):
        if isinstance(file_name, str):
            file_name = Path(file_name)
        self.file_name = f'video{file_name.suffix}'
        self._data = data
        self._save()
        self.info = VideoInfo(self.file_name)
        name = file_name.stem
        md5 = self._compute_md5()
        self.id = f'{name}_{md5}'
        self._video_capture = cv2.VideoCapture(self.file_name)

    def get_frame(self, frame: int):
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, data = self._video_capture.read()
        if not ret:
            return
        return data

    def save_screenshot(self, frame: int, file_path: str | Path):
        data = self.get_frame(frame)
        image = Image.fromarray(data)
        image.save(file_path)

    def save_fragment(self, start_frame: int, frames_number: int, file_path: str | Path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self._temp_file_path, fourcc, self.info.frame_rate,
            (self.info.frame_width, self.info.frame_height))
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(frames_number):
            _, frame = self._video_capture.read()
            video_writer.write(frame)
        video_writer.release()
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()
        args = f"ffmpeg -y -i ./{self._temp_file_path} -c:v libx264 ./{file_path}".split(" ")
        subprocess.call(args=args)
        Path(self._temp_file_path).unlink()


class Data:
    _index_name: str
    _columns: list[str]
    data: pd.DataFrame

    def __init__(self):
        self._create()

    def _create(self):
        self.data = pd.DataFrame(columns=self._columns)
        self.data.index.name = self._index_name


class Storable(Data):

    _file_path: str

    def __init__(self, gc: GoogleCloud, gcs_folder_path: str, gcs_file_name: str):
        super().__init__()
        folder_path = Path(self._file_path).parent
        if not folder_path.exists():
            folder_path.mkdir()
        self._gc = gc
        if isinstance(gcs_folder_path, str):
            gcs_folder_path = Path(gcs_folder_path)
        self._gcs_file_path = gcs_folder_path / gcs_file_name

    def download_data_from_gcs(self, length: int | None = None) -> bool:
        if not self._gc.download_file(self._gcs_file_path, self._file_path):
            return False
        with open(self._file_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            return False
        if list(data.keys()) != self._columns:
            return False
        if length is not None:
            if len(data[list(data.keys())[0]]) != length:
                return False
        self.data = pd.DataFrame.from_dict(data)
        self.data.index.name = self._index_name
        return True

    def upload_data_to_gcs(self) -> bool:
        data = self.data.to_dict()
        with open(self._file_path, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(data, f)
        return self._gc.upload_file(self._gcs_file_path, self._file_path)


class PredictionsBase(Storable):
    """
    Prediction base.
    """

    _gc_folder_path: str
    _index_name = 'step'
    _columns = ['prediction', 'filtered_prediction', 'time']
    _endpoint_index: int
    _default_prediction: list[float]

    _N: int
    _Wn: float

    def __init__(self, video: Video, gc: GoogleCloud):
        super().__init__(gc, self._gc_folder_path, video.id + '.dat')
        self._video = video
        self._faces_number = 0

        # Calculating maximal number of points (steps)
        min_step_frames = math.ceil(st.secrets['settings'].min_time_step * self._video.info.frame_rate)
        self.points_number = math.ceil(self._video.info.frames_number / min_step_frames)
        if self.points_number > st.secrets['settings'].max_points_number:
            self.points_number = st.secrets['settings'].max_points_number

        # Calculating number of frames between neighbour points (per step)
        self.step_frames = math.ceil(self._video.info.frames_number / self.points_number)

        # Calculating time between neighbour points (per step)
        self.step_time = self.step_frames / self._video.info.frame_rate

        # Calculating number of points
        self.points_number = math.ceil(self._video.info.frames_number / self.step_frames)

        # Downloading data from GCS
        if not super().download_data_from_gcs(length=self.points_number):

            # Recognizing emotions arousal
            self._predicts()

            # Filtering emotions arousal
            self._filter()

            # Uploading data to GCS
            super().upload_data_to_gcs()

    def _predicts(self):
        """Recognize valence and arousal of emotion in each video frame."""

        def _extract_face(frame, detector):
            """Extracts a face from the frame."""
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = detector.detectMultiScale(
                gray_frame, st.secrets['face-detector'].scale_factor,
                st.secrets['face-detector'].min_neighbors, 0,
                (st.secrets['face-detector'].min_face_size, st.secrets['face-detector'].min_face_size)
            )
            if len(face_rects) == 0:
                return
            x = y = w = h = 0
            for face_rect in face_rects:
                if face_rect[2] > w and face_rect[3] > h:
                    x, y, w, h = face_rect
            face = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            return np.asarray(face).tolist()

        # Creating face detector
        haar_file = (Path(cv2.__file__).parent / 'data' / st.secrets['face-detector'].haar_file_name).as_posix()
        face_detector = cv2.CascadeClassifier(haar_file)

        # Creating video capture
        video_capture = cv2.VideoCapture(self._video.file_name)

        # Showing progress bar
        empty = st.empty()
        progress_bar = empty.progress(0.0)
        start_time = datetime.now()

        prediction = self._default_prediction
        values = []
        fails = []
        for step_index in range(self.points_number):
            # Retrieving video frame at point
            frame_index = step_index * self.step_frames
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, image = video_capture.read()
            if ret:
                # Extracting face at frame
                face_image = _extract_face(image, face_detector)
                # Recognizing emotion arousal
                if face_image is not None:
                    self._faces_number += 1
                    prediction = self._gc.endpoints[self._endpoint_index].predict(
                        instances=[face_image],
                        use_dedicated_endpoint=True
                    ).predictions[0]
            else:
                fails.append(step_index)

            # Adding to result list
            values.append(prediction)

            # Updating progress bar
            current_time = datetime.now()
            elapsed_time = current_time - start_time
            elapsed_time_str = str(elapsed_time).split('.')[0]
            mean_speed = elapsed_time / (step_index + 1)
            left_time = mean_speed * (self.points_number - step_index - 1)
            left_time_str = str(left_time).split('.')[0]
            percent = (step_index + 1) / self.points_number
            progress_bar.progress(
               percent,
                f'{step_index + 1}/{self.points_number} ({percent:.0%}) '
                f'[Elapsed time: {elapsed_time_str}, Left time: {left_time_str}]'
            )

        # Releasing video capture
        video_capture.release()

        # Hiding progress bar
        empty.empty()

        # Creating data
        self.data['prediction'] = values
        today = datetime.today()
        start_date = datetime(today.year, today.month, today.day)
        end_date = start_date + timedelta(seconds=self.step_time * (self.points_number - 1))
        self.data['time'] = pd.date_range(start=start_date, end=end_date, periods=self.points_number)

    def _filter(self):
        raw = pd.DataFrame(self.data['prediction'].to_list()).to_numpy()

        # Applying median filter
        x = medfilt2d(raw, kernel_size=[3, 1])

        # Applying Butterworth filter
        # noinspection PyTupleAssignmentBalance
        b, a = butter(self._N, self._Wn)
        y = filtfilt(b, a, x, axis=0)

        self.data['filtered_prediction'] = y.tolist()


class Predictions(PredictionsBase):
    """
    Emotion names in processed video
    """

    _gc_folder_path = 'classification/presentation/video/predictions'
    _file_path = 'data/predictions.dat'
    _columns = [
        'emotion', 'probability', 'filtered_emotion', 'filtered_probability'
    ] + PredictionsBase._columns
    _endpoint_index = 0
    _default_prediction = [1. / len(st.session_state.emotions)] * len(st.session_state.emotions)

    _N = 2
    _Wn = 0.2

    def _predicts(self):
        super()._predicts()
        self.data['emotion'] = self.data['prediction'].map(
            lambda p: list(st.session_state.emotions)[p.index(max(p))])
        self.data['probability'] = self.data['prediction'].map(max)

    def _filter(self):
        super()._filter()
        self.data['filtered_emotion'] = self.data['filtered_prediction'].map(
            lambda p: list(st.session_state.emotions)[p.index(max(p))])
        self.data['filtered_probability'] = self.data['filtered_prediction'].map(max)


class PredictionsVa(PredictionsBase):
    """
    Emotions arousal in processed video
    """

    _gc_folder_path = 'va/presentation/video/predictions'
    _file_path = 'data/predictions.dat'
    _columns = [
        'arousal', 'valence', 'filtered_arousal', 'filtered_valence'
    ] + PredictionsBase._columns
    _endpoint_index = 1
    _default_prediction = [0., 0.]

    _N = 2
    _Wn = 0.2

    def _predicts(self):
        super()._predicts()
        self.data['valence'] = self.data['prediction'].map(lambda p: p[0])
        self.data['arousal'] = self.data['prediction'].map(lambda p: p[1])

    def _filter(self):
        super()._filter()
        self.data['filtered_valence'] = self.data['filtered_prediction'].map(lambda p: p[0])
        self.data['filtered_arousal'] = self.data['filtered_prediction'].map(lambda p: p[1])


class HyperParamsBase(Storable):
    """
    Hyperparameters for searching fragments to trailer creation
    """

    _gc_folder_path: str
    _columns = ['title',
                'low_limit', 'high_limit', 'step', 'possible_options',
                'init_low_bound', 'init_high_bound', 'init_options',
                'low_bound', 'high_bound', 'options']
    _index_name = 'name'
    _names: list[str]
    _titles: list[str]

    def __init__(self, video: Video, predictions: PredictionsBase, gc: GoogleCloud):
        super().__init__(gc, self._gc_folder_path, video.id + '.dat')
        self._video = video
        self._predictions = predictions

        # Downloading data from GCS
        if not super().download_data_from_gcs(length=len(self._names)):
            self._init()

        # Uploading data to GCS
        super().upload_data_to_gcs()

    def _init(self):
        self.data['title'] = self._titles
        self.data.index = self._names

    @property
    def names(self):
        return self._names

    @property
    def titles(self):
        return self._titles

    def get_limits(self, name: str) -> (float, float):
        low, high = self.data.loc[name, ['low_limit', 'high_limit']]
        return low, high

    def get_possible_options(self, name: str) -> (float, float):
        options = self.data.at[name, 'possible_options']
        return options

    def get_init_bounds_and_step(self, name: str) -> (float, float):
        low, high, step = self.data.loc[name, ['init_low_bound', 'init_high_bound', 'step']]
        return low, high, step

    def get_init_options(self, name: str) -> (float, float):
        options = self.data.at[name, 'init_options']
        return options

    def get_bounds(self, name: str) -> (float, float | None):
        low, high = self.data.loc[name, ['low_bound', 'high_bound']]
        return low, high

    def get_options(self, name: str) -> (float, float | None):
        options = self.data.at[name, 'options']
        return options

    def set_bounds(self, name: str, low: float, high: float | None):
        self.data.loc[name, ['low_bound', 'high_bound']] = low, high
        super().upload_data_to_gcs()

    def set_options(self, name: str, options: list):
        self.data.at[name, 'options'] = options
        super().upload_data_to_gcs()


class HyperParams(HyperParamsBase):
    """
    Hyperparameters for searching fragments to trailer creation
    """

    _gc_folder_path = 'classification/presentation/video/hyperparams'
    _file_path = 'data/hyperparams.dat'
    _names = [
        'emotions',
        'min_probability',
        'fragment_duration',
        'min_time_between_fragments'
    ]
    _titles = [
        'Emotions',
        'Min. probability',
        'Fragment duration [s]',
        'Min. time between fragments [s]'
    ]

    def __init__(self, video: Video, predictions: Predictions, gc: GoogleCloud):
        super().__init__(video, predictions, gc)

    def _init(self):
        super()._init()

        self.data.at['emotions', 'possible_options'] = self._predictions.data['emotion'].unique().tolist()
        self.data.at['emotions', 'init_options'] = self.data.at['emotions', 'possible_options']
        self.data.at['emotions', 'options'] = self.data.at['emotions', 'possible_options']

        self.data.at['min_probability', 'low_limit'] = self._predictions.data['filtered_probability'].min()
        self.data.at['min_probability', 'high_limit'] = self._predictions.data['filtered_probability'].max()
        self.data.at['min_probability', 'step'] = min(
            self.data.at['min_probability', 'high_limit'] - self.data.at['min_probability', 'low_limit'], 0.01
        )
        self.data.at['min_probability', 'init_low_bound'] = self.data.at['min_probability', 'low_limit']
        self.data.at['min_probability', 'low_bound'] = self.data.at['min_probability', 'init_low_bound']

        self.data.at['fragment_duration', 'low_limit'] = min(self._video.info.duration.total_seconds(), 0.)
        self.data.at['fragment_duration', 'high_limit'] = min(self._video.info.duration.total_seconds(), 15.)
        self.data.at['fragment_duration', 'step'] = min(
            self.data.at['fragment_duration', 'high_limit'] -
            self.data.at['fragment_duration', 'low_limit'],
            0.1
        )
        self.data.at['fragment_duration', 'init_low_bound'] = self.data.at['fragment_duration', 'low_limit']
        self.data.at['fragment_duration', 'init_high_bound'] = self.data.at['fragment_duration', 'high_limit']
        self.data.at['fragment_duration', 'low_bound'] = self.data.at['fragment_duration', 'init_low_bound']
        self.data.at['fragment_duration', 'high_bound'] = self.data.at['fragment_duration', 'init_low_bound']

        self.data.at['min_time_between_fragments', 'low_limit'] = min(self._video.info.duration.total_seconds(), 0.)
        self.data.at['min_time_between_fragments', 'high_limit'] = min(self._video.info.duration.total_seconds(), 15.)
        self.data.at['min_time_between_fragments', 'step'] = min(
            self.data.at['min_time_between_fragments', 'high_limit'] -
            self.data.at['min_time_between_fragments', 'low_limit'],
            0.1
        )
        self.data.at['min_time_between_fragments', 'init_low_bound'] = (
                self.data.at['min_time_between_fragments', 'high_limit'] -
                self.data.at['min_time_between_fragments', 'high_limit']
        ) / 2
        self.data.at['min_time_between_fragments', 'low_bound'] = self.data.at[
            'min_time_between_fragments', 'init_low_bound']


class HyperParamsVa(HyperParamsBase):
    """
    Hyperparameters for searching fragments to trailer creation
    """

    _gc_folder_path = 'va/presentation/video/hyperparams'
    _file_path = 'data/hyperparams_va.dat'
    _names = [
       'emotion_valence',
       'emotion_arousal',
       'fragment_duration',
       'min_time_between_fragments'
    ]
    _titles = [
        'Emotion valence',
        'Emotion arousal',
        'Fragment duration [s]',
        'Min. time between fragments [s]'
    ]

    def __init__(self, video: Video, predictions: PredictionsVa, gc: GoogleCloud):
        super().__init__(video, predictions, gc)

    def _init(self):
        super()._init()

        self.data.at['emotion_valence', 'low_limit'] = self._predictions.data['filtered_valence'].min()
        self.data.at['emotion_valence', 'high_limit'] = self._predictions.data['filtered_valence'].max()
        self.data.at['emotion_valence', 'step'] = min(
            self.data.at['emotion_valence', 'high_limit'] -
            self.data.at['emotion_valence', 'low_limit'],
            0.01
        )
        self.data.at['emotion_valence', 'init_low_bound'] = self.data.at['emotion_valence', 'low_limit']
        self.data.at['emotion_valence', 'init_high_bound'] = self.data.at['emotion_valence', 'high_limit']
        self.data.at['emotion_valence', 'low_bound'] = self.data.at['emotion_valence', 'init_low_bound']
        self.data.at['emotion_valence', 'high_bound'] = self.data.at['emotion_valence', 'init_high_bound']

        self.data.at['emotion_arousal', 'low_limit'] = self._predictions.data['filtered_arousal'].min()
        self.data.at['emotion_arousal', 'high_limit'] = self._predictions.data['filtered_arousal'].max()
        self.data.at['emotion_arousal', 'step'] = min(
            self.data.at['emotion_arousal', 'high_limit'] -
            self.data.at['emotion_arousal', 'low_limit'],
            0.01
        )
        self.data.at['emotion_arousal', 'init_low_bound'] = self.data.at['emotion_arousal', 'low_limit']
        self.data.at['emotion_arousal', 'init_high_bound'] = self.data.at['emotion_arousal', 'high_limit']
        self.data.at['emotion_arousal', 'low_bound'] = self.data.at['emotion_arousal', 'init_low_bound']
        self.data.at['emotion_arousal', 'high_bound'] = self.data.at['emotion_arousal', 'init_high_bound']

        self.data.at['fragment_duration', 'low_limit'] = min(self._video.info.duration.total_seconds(), 0.)
        self.data.at['fragment_duration', 'high_limit'] = min(self._video.info.duration.total_seconds(), 15.)
        self.data.at['fragment_duration', 'step'] = min(
            self.data.at['fragment_duration', 'high_limit'] -
            self.data.at['fragment_duration', 'low_limit'],
            0.1
        )
        self.data.at['fragment_duration', 'init_low_bound'] = self.data.at['fragment_duration', 'low_limit']
        self.data.at['fragment_duration', 'init_high_bound'] = self.data.at['fragment_duration', 'high_limit']
        self.data.at['fragment_duration', 'low_bound'] = self.data.at['fragment_duration', 'init_low_bound']
        self.data.at['fragment_duration', 'high_bound'] = self.data.at['fragment_duration', 'init_high_bound']

        self.data.at['min_time_between_fragments', 'low_limit'] = min(self._video.info.duration.total_seconds(), 0.)
        self.data.at['min_time_between_fragments', 'high_limit'] = min(self._video.info.duration.total_seconds(), 15.)
        self.data.at['min_time_between_fragments', 'step'] = min(
            self.data.at['min_time_between_fragments', 'high_limit'] -
            self.data.at['min_time_between_fragments', 'low_limit'],
            0.1
        )
        self.data.at['min_time_between_fragments', 'init_low_bound'] = (
            self.data.at['min_time_between_fragments', 'high_limit'] -
            self.data.at['min_time_between_fragments', 'high_limit']
        ) / 2
        self.data.at['min_time_between_fragments', 'low_bound'] = self.data.at[
            'min_time_between_fragments', 'init_low_bound']


class FragmentsBase(Storable):

    _gc_folder_path: str
    _columns = [
        'start_step', 'end_step', 'steps',
        'start', 'peak', 'end', 'time',
        'start_arousal', 'peak_arousal', 'end_arousal',
    ]
    _index_name = 'fragment'

    def __init__(self, video: Video, predictions: PredictionsBase, hyperparams: HyperParamsBase, gc: GoogleCloud):
        super().__init__(gc, 'fragments', video.id + '.dat')
        self._video = video
        self._predictions = predictions
        self._hyperparams = hyperparams

        # Downloading data from GCS
        super().download_data_from_gcs()

    def find_fragments(self):
        # Searching arousal peaks
        peaks, _ = find_peaks(
            self._predictions.data['filtered']
        )
        if len(peaks) == 0:
            return
        super()._create()
        self.data['peak_step'] = peaks
        self.data['peak_arousal'] = self._predictions.data.loc[self.data['peak_step'], 'filtered'].reset_index(drop=True)

        arousal_low_bound, arousal_high_bound = self._hyperparams.get_bounds('emotion_arousal')

        # Excluding fragments which peak arousal is out of bounds
        self.data = self.data.loc[
            self.data['peak_arousal'].between(arousal_low_bound, arousal_high_bound, inclusive='both')
        ]
        self.data.reset_index(drop=True, inplace=True)
        self.data.index.name = self._index_name
        if self.data.shape[0] == 0:
            return

        # Looking for fragment bounds
        self.data['prev_peak_step'] = -1, *self.data['peak_step'].iloc[:-1]
        self.data['next_peak_step'] = *self.data['peak_step'].iloc[1:], *[self._predictions.data.index[-1] + 1]
        valid_arousals = self._predictions.data['filtered'].between(
            arousal_low_bound, arousal_high_bound, inclusive='both')
        for fragment, (prev_step, step, next_step) in self.data[
            ['prev_peak_step', 'peak_step', 'next_peak_step']].iterrows():
            left_part = valid_arousals.loc[prev_step + 1: step - 1].sort_index(ascending=False)
            if sum(left_part) == 0:
                start = step
            elif sum(left_part) == len(left_part):
                start = left_part.index[0]
            else:
                start = left_part.idxmin() + 1
            right_part = valid_arousals.loc[step + 1: next_step - 1]
            if sum(right_part) == 0:
                end = step
            elif sum(right_part) == len(right_part):
                end = right_part.index[-1]
            else:
                end = right_part.idxmin() - 1
            self.data.loc[fragment, ['start_step', 'end_step']] = start, end

        # Union overlapped fragments
        self.data['next_start_step'] = *self.data['start_step'].iloc[1:], self._predictions.points_number + 1
        while True:
            if self.data.shape[0] <= 1:
                break
            overlapped_fragment_indices = self.data.loc[self.data['end_step'] >= self.data['next_start_step']].index
            if overlapped_fragment_indices.empty:
                break
            first_index = overlapped_fragment_indices[0]
            second_index = first_index + 1
            self.data.loc[first_index, ['end_step', 'next_peak_step', 'next_start_step']] = (
                self.data.loc[second_index, ['end_step', 'next_peak_step', 'next_start_step']]
            )
            if self.data.at[first_index, 'peak_arousal'] < self.data.at[second_index, 'peak_arousal']:
                self.data.loc[first_index, ['peak_step', 'peak_arousal']] = (
                    self.data.loc[second_index, ['peak_step', 'peak_arousal']]
                )
            self.data.drop(index=second_index, inplace=True)
            self.data.reset_index(drop=True, inplace=True, names=self._index_name)
            self.data.index.name = self._index_name

        # Calculate fragment steps
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1

        duration_low_bound, duration_high_bound = self._hyperparams.get_bounds('fragment_duration')

        # Excluding too short fragments
        min_steps = math.ceil(duration_low_bound / self._predictions.step_time)
        short_fragment_indices = self.data.loc[self.data['steps'] < min_steps].index
        if not short_fragment_indices.empty:
            self.data.drop(short_fragment_indices, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            self.data.index.name = self._index_name

        # Cutting too long fragments
        max_steps = math.floor(duration_high_bound / self._predictions.step_time)
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1
        long_fragments = self.data.loc[self.data['steps'] > max_steps, ['start_step', 'peak_step', 'end_step', 'steps']]
        if not long_fragments.empty:
            for index, (start_step, peak_step, end_step, steps) in long_fragments.iterrows():
                left_steps = peak_step - start_step + 1
                # right_steps = steps - left_steps
                extra_steps = steps - max_steps
                left_extra_steps = math.ceil(extra_steps * left_steps / steps)
                right_extra_steps = extra_steps - left_extra_steps
                start_step += left_extra_steps
                end_step -= right_extra_steps
                steps -= extra_steps
                self.data.loc[index, ['start_step', 'end_step', 'steps']] = start_step, end_step, steps

        if not self.data.empty:

            # Thinning fragments
            min_time, _ = self._hyperparams.get_bounds('min_time_between_fragments')
            min_steps = math.ceil(min_time / self._predictions.step_time)
            while True:
                self.data['next_start_step'] = (
                    *self.data['start_step'].iloc[1:], self._predictions.points_number + min_steps)
                self.data['steps_to_next'] = self.data['next_start_step'] - self.data['end_step']
                if self.data.shape[0] <= 1:
                    break
                closed_fragments = self.data.loc[self.data['steps_to_next'] < min_steps].index
                if closed_fragments.empty:
                    break
                first_index = closed_fragments[0]
                second_index = first_index + 1
                if self.data.at[first_index, 'peak_arousal'] < self.data.at[second_index, 'peak_arousal']:
                    dropping_index = first_index
                else:
                    dropping_index = second_index
                self.data.drop(index=dropping_index, inplace=True)
                self.data.reset_index(drop=True, inplace=True, names=self._index_name)
                self.data.index.name = self._index_name

            # Retrieving arousals at bounds
            self.data['start_arousal'] = self._predictions.data.loc[
                self.data['start_step'], 'filtered'].reset_index(drop=True)
            self.data['end_arousal'] = self._predictions.data.loc[
                self.data['end_step'], 'filtered'].reset_index(drop=True)

            # Retrieving time at peaks and bounds
            self.data['start'] = self._predictions.data.loc[
                self.data['start_step'], 'time'].reset_index(drop=True)
            self.data['peak'] = self._predictions.data.loc[
                self.data['peak_step'], 'time'].reset_index(drop=True)
            self.data['end'] = self._predictions.data.loc[
                self.data['end_step'], 'time'].reset_index(drop=True)
            self.data['end'] += pd.to_timedelta(self._predictions.step_time, unit='s')
            self.data['time'] = self.data['end'] - self.data['start']

        # Remove helper fields
        self.data = self.data[self._columns]
        self.data = self.data.convert_dtypes()

        # Saving data to GCS
        super().upload_data_to_gcs()


class Fragments(FragmentsBase):

    _gc_folder_path = 'classification/presentation/video/fragments'
    _file_path = 'data/fragments.dat'
    _columns = [
        'emotion', 'start_step', 'peak_step', 'end_step', 'steps',
        'start', 'peak', 'end', 'time',
        'start_probability', 'peak_probability', 'end_probability',
    ]
    _index_name = 'fragment'

    def __init__(self, video: Video, predictions: Predictions, hyperparams: HyperParams, gc: GoogleCloud):
        super().__init__(video, predictions, hyperparams, gc)

    def find_fragments(self):

        # Excluding
        emotions = self._predictions.data['filtered_emotion'].copy()
        probabilities = self._predictions.data['filtered_probability'].copy()
        times = self._predictions.data['time']
        selected_emotions = self._hyperparams.get_options('emotions')
        is_emotions_selected = emotions.isin(selected_emotions)
        emotions.loc[~is_emotions_selected] = -1
        probabilities.loc[~is_emotions_selected] = -1

        # Searching probability peaks
        peaks, _ = find_peaks(probabilities)
        if len(peaks) == 0:
            return
        super()._create()
        self.data['peak_step'] = peaks
        self.data['emotion'] = emotions.loc[self.data['peak_step']].reset_index(drop=True)
        self.data['peak_probability'] = probabilities.loc[self.data['peak_step']].reset_index(drop=True)

        probability_low_bound, _ = self._hyperparams.get_bounds('min_probability')

        # Excluding fragments which peak probability is out of bounds
        self.data = self.data.loc[self.data['peak_probability'] >= probability_low_bound]
        self.data.reset_index(drop=True, inplace=True)
        self.data.index.name = self._index_name
        if self.data.shape[0] == 0:
            return

        # Looking for fragment bounds
        self.data['prev_peak_step'] = -1, *self.data['peak_step'].iloc[:-1]
        self.data['next_peak_step'] = *self.data['peak_step'].iloc[1:], *[self._predictions.data.index[-1] + 1]
        for fragment, (emotion, prev_step, step, next_step) in self.data[
            ['emotion', 'prev_peak_step', 'peak_step', 'next_peak_step']].iterrows():
            left_part = (
                (probabilities.loc[prev_step + 1: step - 1] >= probability_low_bound) &
                (emotions.loc[prev_step + 1: step - 1] == emotion)
            ).sort_index(ascending=False)
            if sum(left_part) == 0:
                start = step
            elif sum(left_part) == len(left_part):
                start = left_part.index[-1]
            else:
                start = left_part.idxmin() + 1
            right_part = (
                (probabilities.loc[step + 1: next_step - 1] >= probability_low_bound) &
                (emotions.loc[step + 1: next_step - 1] == emotion)
            )
            if sum(right_part) == 0:
                end = step
            elif sum(right_part) == len(right_part):
                end = right_part.index[-1]
            else:
                end = right_part.idxmin() - 1
            self.data.loc[fragment, ['start_step', 'end_step']] = start, end

        # Union overlapped fragments
        self.data['next_start_step'] = *self.data['start_step'].iloc[1:], self._predictions.points_number + 1
        self.data['next_emotion'] = *self.data['emotion'].iloc[1:], ''
        while True:
            if self.data.shape[0] <= 1:
                break
            overlapped_fragment_indices = self.data.loc[
                (self.data['end_step'] >= self.data['next_start_step']) &
                (self.data['emotion'] == self.data['next_emotion'])
            ].index
            if overlapped_fragment_indices.empty:
                break
            first_index = overlapped_fragment_indices[0]
            second_index = first_index + 1
            self.data.loc[first_index, ['end_step', 'next_peak_step', 'next_start_step']] = (
                self.data.loc[second_index, ['end_step', 'next_peak_step', 'next_start_step']]
            )
            if self.data.at[first_index, 'peak_probability'] < self.data.at[second_index, 'peak_probability']:
                self.data.loc[first_index, ['peak_step', 'peak_probability']] = (
                    self.data.loc[second_index, ['peak_step', 'peak_probability']]
                )
            self.data.drop(index=second_index, inplace=True)
            self.data.reset_index(drop=True, inplace=True, names=self._index_name)
            self.data.index.name = self._index_name

        # Calculate fragment steps
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1

        duration_low_bound, duration_high_bound = self._hyperparams.get_bounds('fragment_duration')

        # Excluding too short fragments
        min_steps = math.ceil(duration_low_bound / self._predictions.step_time)
        short_fragment_indices = self.data.loc[self.data['steps'] < min_steps].index
        if not short_fragment_indices.empty:
            self.data.drop(short_fragment_indices, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            self.data.index.name = self._index_name

        # Cutting too long fragments
        max_steps = math.floor(duration_high_bound / self._predictions.step_time)
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1
        long_fragments = self.data.loc[self.data['steps'] > max_steps, ['start_step', 'peak_step', 'end_step', 'steps']]
        if not long_fragments.empty:
            for index, (start_step, peak_step, end_step, steps) in long_fragments.iterrows():
                left_steps = peak_step - start_step + 1
                # right_steps = steps - left_steps
                extra_steps = steps - max_steps
                left_extra_steps = math.ceil(extra_steps * left_steps / steps)
                right_extra_steps = extra_steps - left_extra_steps
                start_step += left_extra_steps
                end_step -= right_extra_steps
                steps -= extra_steps
                self.data.loc[index, ['start_step', 'end_step', 'steps']] = start_step, end_step, steps

        if not self.data.empty:

            # Thinning fragments
            min_time, _ = self._hyperparams.get_bounds('min_time_between_fragments')
            min_steps = math.ceil(min_time / self._predictions.step_time)
            while True:
                self.data['next_start_step'] = (
                    *self.data['start_step'].iloc[1:], self._predictions.points_number + min_steps)
                self.data['steps_to_next'] = self.data['next_start_step'] - self.data['end_step']
                if self.data.shape[0] <= 1:
                    break
                closed_fragments = self.data.loc[self.data['steps_to_next'] < min_steps].index
                if closed_fragments.empty:
                    break
                first_index = closed_fragments[0]
                second_index = first_index + 1
                if self.data.at[first_index, 'peak_probability'] < self.data.at[second_index, 'peak_probability']:
                    dropping_index = first_index
                else:
                    dropping_index = second_index
                self.data.drop(index=dropping_index, inplace=True)
                self.data.reset_index(drop=True, inplace=True, names=self._index_name)
                self.data.index.name = self._index_name

            # Retrieving arousals at bounds
            self.data['start_probability'] = probabilities.loc[self.data['start_step']].reset_index(drop=True)
            self.data['end_probability'] = probabilities.loc[self.data['end_step']].reset_index(drop=True)

            # Retrieving time at peaks and bounds
            self.data['start'] = times.loc[self.data['start_step']].reset_index(drop=True)
            self.data['peak'] = times.loc[self.data['peak_step']].reset_index(drop=True)
            self.data['end'] = times.loc[self.data['end_step']].reset_index(drop=True)
            self.data['end'] += pd.to_timedelta(self._predictions.step_time, unit='s')
            self.data['time'] = self.data['end'] - self.data['start']

        # Remove helper fields
        self.data = self.data[self._columns]
        self.data = self.data.convert_dtypes()

        # Saving data to GCS
        super().upload_data_to_gcs()


class FragmentsVa(FragmentsBase):

    _gc_folder_path = 'va/presentation/video/fragments'
    _file_path = 'data/fragments_va.dat'
    _columns = [
        'start_step', 'peak_step', 'end_step', 'steps',
        'start', 'peak', 'end', 'time',
        'start_arousal', 'peak_arousal', 'end_arousal',
    ]
    _index_name = 'fragment'

    def __init__(self, video: Video, predictions: PredictionsVa, hyperparams: HyperParamsVa, gc: GoogleCloud):
        super().__init__(video, predictions, hyperparams, gc)

    def find_fragments(self):

        # Excluding points with valence value is out of bounds
        arousals = self._predictions.data['filtered_arousal'].copy()
        valences = self._predictions.data['filtered_valence'].copy()
        valence_low_bound, valence_high_bound = self._hyperparams.get_bounds('emotion_valence')
        arousals.loc[
            ~valences.between(valence_low_bound, valence_high_bound, inclusive='both')
        ] = -2

        # Searching arousal peaks
        peaks, _ = find_peaks(arousals)
        if len(peaks) == 0:
            return
        super()._create()
        self.data['peak_step'] = peaks
        self.data['peak_arousal'] = arousals.loc[self.data['peak_step']].to_list()

        arousal_low_bound, arousal_high_bound = self._hyperparams.get_bounds('emotion_arousal')

        # Excluding fragments which peak arousal is out of bounds
        self.data = self.data.loc[
            self.data['peak_arousal'].between(arousal_low_bound, arousal_high_bound, inclusive='both')
        ]
        self.data.reset_index(drop=True, inplace=True)
        self.data.index.name = self._index_name
        if self.data.shape[0] == 0:
            return

        # Looking for fragment bounds
        self.data['prev_peak_step'] = -1, *self.data['peak_step'].iloc[:-1]
        self.data['next_peak_step'] = *self.data['peak_step'].iloc[1:], *[self._predictions.data.index[-1] + 1]
        valid_arousals = arousals.between(
            arousal_low_bound, arousal_high_bound, inclusive='both')
        for fragment, (prev_step, step, next_step) in self.data[
            ['prev_peak_step', 'peak_step', 'next_peak_step']].iterrows():
            left_part = valid_arousals.loc[prev_step + 1: step - 1].sort_index(ascending=False)
            if sum(left_part) == 0:
                start = step
            elif sum(left_part) == len(left_part):
                start = left_part.index[-1]
            else:
                start = left_part.idxmin() + 1
            right_part = valid_arousals.loc[step + 1: next_step - 1]
            if sum(right_part) == 0:
                end = step
            elif sum(right_part) == len(right_part):
                end = right_part.index[-1]
            else:
                end = right_part.idxmin() - 1
            self.data.loc[fragment, ['start_step', 'end_step']] = start, end

        # Union overlapped fragments
        self.data['next_start_step'] = *self.data['start_step'].iloc[1:], self._predictions.points_number + 1
        while True:
            if self.data.shape[0] <= 1:
                break
            overlapped_fragment_indices = self.data.loc[self.data['end_step'] >= self.data['next_start_step']].index
            if overlapped_fragment_indices.empty:
                break
            first_index = overlapped_fragment_indices[0]
            second_index = first_index + 1
            self.data.loc[first_index, ['end_step', 'next_peak_step', 'next_start_step']] = (
                self.data.loc[second_index, ['end_step', 'next_peak_step', 'next_start_step']]
            )
            if self.data.at[first_index, 'peak_arousal'] < self.data.at[second_index, 'peak_arousal']:
                self.data.loc[first_index, ['peak_step', 'peak_arousal']] = (
                    self.data.loc[second_index, ['peak_step', 'peak_arousal']]
                )
            self.data.drop(index=second_index, inplace=True)
            self.data.reset_index(drop=True, inplace=True, names=self._index_name)
            self.data.index.name = self._index_name

        # Calculate fragment steps
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1

        duration_low_bound, duration_high_bound = self._hyperparams.get_bounds('fragment_duration')

        # Excluding too short fragments
        min_steps = math.ceil(duration_low_bound / self._predictions.step_time)
        short_fragment_indices = self.data.loc[self.data['steps'] < min_steps].index
        if not short_fragment_indices.empty:
            self.data.drop(short_fragment_indices, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            self.data.index.name = self._index_name

        # Cutting too long fragments
        max_steps = math.floor(duration_high_bound / self._predictions.step_time)
        self.data['steps'] = self.data['end_step'] - self.data['start_step'] + 1
        long_fragments = self.data.loc[self.data['steps'] > max_steps, ['start_step', 'peak_step', 'end_step', 'steps']]
        if not long_fragments.empty:
            for index, (start_step, peak_step, end_step, steps) in long_fragments.iterrows():
                left_steps = peak_step - start_step + 1
                # right_steps = steps - left_steps
                extra_steps = steps - max_steps
                left_extra_steps = math.ceil(extra_steps * left_steps / steps)
                right_extra_steps = extra_steps - left_extra_steps
                start_step += left_extra_steps
                end_step -= right_extra_steps
                steps -= extra_steps
                self.data.loc[index, ['start_step', 'end_step', 'steps']] = start_step, end_step, steps

        if not self.data.empty:

            # Thinning fragments
            min_time, _ = self._hyperparams.get_bounds('min_time_between_fragments')
            min_steps = math.ceil(min_time / self._predictions.step_time)
            while True:
                self.data['next_start_step'] = (
                    *self.data['start_step'].iloc[1:], self._predictions.points_number + min_steps)
                self.data['steps_to_next'] = self.data['next_start_step'] - self.data['end_step']
                if self.data.shape[0] <= 1:
                    break
                closed_fragments = self.data.loc[self.data['steps_to_next'] < min_steps].index
                if closed_fragments.empty:
                    break
                first_index = closed_fragments[0]
                second_index = first_index + 1
                if self.data.at[first_index, 'peak_arousal'] < self.data.at[second_index, 'peak_arousal']:
                    dropping_index = first_index
                else:
                    dropping_index = second_index
                self.data.drop(index=dropping_index, inplace=True)
                self.data.reset_index(drop=True, inplace=True, names=self._index_name)
                self.data.index.name = self._index_name

            # Retrieving arousals at bounds
            self.data['start_arousal'] = self._predictions.data.loc[
                self.data['start_step'], 'filtered_arousal'].reset_index(drop=True)
            self.data['end_arousal'] = self._predictions.data.loc[
                self.data['end_step'], 'filtered_arousal'].reset_index(drop=True)

            # Retrieving time at peaks and bounds
            self.data['start'] = self._predictions.data.loc[
                self.data['start_step'], 'time'].reset_index(drop=True)
            self.data['peak'] = self._predictions.data.loc[
                self.data['peak_step'], 'time'].reset_index(drop=True)
            self.data['end'] = self._predictions.data.loc[
                self.data['end_step'], 'time'].reset_index(drop=True)
            self.data['end'] += pd.to_timedelta(self._predictions.step_time, unit='s')
            self.data['time'] = self.data['end'] - self.data['start']

        # Remove helper fields
        self.data = self.data[self._columns]
        self.data = self.data.convert_dtypes()

        # Saving data to GCS
        super().upload_data_to_gcs()


class TrailerBase(Storable):
    _gcs_screenshots_path: str
    _screenshots_path: str
    _columns = [
        'screenshot_frame', 'screenshot_file_path',
        'screenshot_gcs_file_path', 'screenshot_url',
        'fragment_start_frame', 'fragment_frames_number',
        'selected'
    ]
    _index_name = 'fragment'
    trailer_file_path: str

    def __init__(self, video: Video, predictions: PredictionsBase, fragments: FragmentsBase, gc: GoogleCloud):
        super().__init__(gc, 'trailer', video.id + '.dat')
        self._video = video
        self._predictions = predictions
        self._fragments = fragments

        path = Path(self._screenshots_path)
        if not path.exists():
            path.mkdir()

        # Downloading data from GCS
        if super().download_data_from_gcs(length=self._fragments.data.shape[0]):
            # Saving screenshots and fragments
            self._save_screenshots()
        else:
            # Creating data
            self.create_data()

    def _save_screenshots(self):
        for _, (frame, file_path, gcs_file_path) in self.data[
            ['screenshot_frame', 'screenshot_file_path', 'screenshot_gcs_file_path']].iterrows():
            self._video.save_screenshot(frame, file_path)
            self._gc.upload_file(gcs_file_path, file_path)

    def create_data(self):
        # Creating data
        super()._create()
        self.data['screenshot_frame'] = self._fragments.data['peak_step'] * self._predictions.step_frames
        self.data['screenshot_file_path'] = self._fragments.data.index.map(
            lambda fragment: f'{self._screenshots_path}/fragment_{fragment}.jpg'
        )
        self.data['screenshot_gcs_file_path'] = self._fragments.data.index.map(
            lambda fragment: f'{self._gcs_screenshots_path}/fragment_{fragment}.jpg'
        )
        self.data['screenshot_url'] = self.data['screenshot_file_path'].map(
            lambda file_path: f'https://storage.cloud.google.com/{st.secrets["gc-storage"].bucket_id}/'
                              f'{self._gcs_screenshots_path}/{Path(file_path).name}')
        self.data['fragment_start_frame'] = self._fragments.data['start_step'] * self._predictions.step_frames
        self.data['fragment_frames_number'] = self._fragments.data['steps'] * self._predictions.step_frames
        self.data['selected'] = False

        # Saving data to GCS
        super().upload_data_to_gcs()

        # Saving screenshots and fragments
        self._save_screenshots()
        # self._save_fragments()

    def select_fragments(self, fragments: list[int]):
        self.data['selected'] = False
        self.data.loc[fragments, 'selected'] = True

    @property
    def selected_fragments(self):
        return self.data.loc[self.data['selected']].index.to_list()

    def save(self):
        """Saves the trailer to a file."""
        temp_name = 'temp.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        trailer_writer = cv2.VideoWriter(
            temp_name, fourcc, self._video.info.frame_rate,
            (self._video.info.frame_width, self._video.info.frame_height))
        data = self.data.loc[self.data['selected'], ['fragment_start_frame', 'fragment_frames_number']]
        trailer_frames_number = data['fragment_frames_number'].sum()
        frames_number = 0
        empty = st.empty()
        progress_bar = empty.progress(0.0)
        start_time = datetime.now()
        for fragment, (fragment_start_frame, fragment_frames_number) in data.iterrows():
            for frame in range(fragment_start_frame, fragment_start_frame + fragment_frames_number):
                trailer_writer.write(self._video.get_frame(frame))
                frames_number += 1

                # Updating progress bar
                current_time = datetime.now()
                elapsed_time = current_time - start_time
                elapsed_time_str = str(elapsed_time).split('.')[0]
                mean_speed = elapsed_time / frames_number
                left_time = mean_speed * (trailer_frames_number - frames_number)
                left_time_str = str(left_time).split('.')[0]
                percent = frames_number / trailer_frames_number
                progress_bar.progress(
                    percent,
                    f'{percent:.0%} [Elapsed time: {elapsed_time_str}, Left time: {left_time_str}]'
                )
        empty.empty()
        trailer_writer.release()
        args = f"ffmpeg -y -i ./{temp_name} -c:v libx264 ./{self.trailer_file_path}".split(" ")
        subprocess.call(args=args)
        Path(temp_name).unlink()


class Trailer(TrailerBase):

    _gcs_screenshots_path = 'classification/presentation/video/screenshots'
    _screenshots_path = 'data/screenshots'
    _file_path = 'data/trailer.dat'
    trailer_file_path = 'data/trailer.mp4'

    def __init__(self, video: Video, predictions: Predictions, fragments: Fragments, gc: GoogleCloud):
        super().__init__(video, predictions, fragments, gc)


class TrailerVa(TrailerBase):

    _gcs_screenshots_path = 'va/presentation/video/screenshots'
    _screenshots_path = 'data/screenshots_va'
    _file_path = 'data/trailer_va.dat'
    trailer_file_name = 'data/trailer_va.mp4'

    def __init__(self, video: Video, predictions: PredictionsVa, fragments: FragmentsVa, gc: GoogleCloud):
        super().__init__(video, predictions, fragments, gc)


def get_help():
    with open('help.md', 'rt') as f:
        md = f.read()
    return md


def on_video_upload():
    st.session_state.video_uploader_changed = True


def upload_video() -> [str, VideoData]:
    """Uploads video file."""
    st.session_state.uploaded_file = st.file_uploader(
        # 'Video file uploading',
        '',
        ['mp4', 'avi', 'mov'],
        # help='Drag and Click the **Browser files** button and select a video file from your storage',
        label_visibility='visible',
        on_change=on_video_upload
    )
    if st.session_state.uploaded_file is None:
        st.session_state.uploaded_file_changed = False


def create_common_components():
    if not st.session_state.video_uploader_changed:
        return
    st.session_state.video_uploader_changed = False
    video = Video(
        st.session_state.uploaded_file.name, st.session_state.uploaded_file.getvalue())
    if 'video_id' in st.session_state:
        if st.session_state.video_id == video.id:
            st.session_state.video_changed = False
            return
    st.session_state.video_changed = True
    st.session_state.video_id = video.id
    st.session_state.video = video


def create_components():
    if not st.session_state.video_changed:
        return
    st.session_state.hyperparams_changed = True
    st.session_state.trailer_saved = False
    with st.spinner('Recognizing emotions...'):
        st.session_state.predictions = Predictions(st.session_state.video, st.session_state.google_cloud)
    st.session_state.hyperparams = HyperParams(
        st.session_state.video, st.session_state.predictions,
        st.session_state.google_cloud)
    st.session_state.fragments = Fragments(
        st.session_state.video, st.session_state.predictions, st.session_state.hyperparams,
        st.session_state.google_cloud)
    st.session_state.trailer = Trailer(
        st.session_state.video, st.session_state.predictions, st.session_state.fragments,
        st.session_state.google_cloud)


def create_components_va():
    if not st.session_state.video_changed:
        return
    st.session_state.hyperparams_va_changed = True
    st.session_state.trailer_va_saved = False
    with st.spinner('Recognizing emotions valence and arousal...'):
        st.session_state.predictions_va = PredictionsVa(st.session_state.video, st.session_state.google_cloud)
    st.session_state.hyperparams_va = HyperParamsVa(
        st.session_state.video, st.session_state.predictions_va,
        st.session_state.google_cloud)
    st.session_state.fragments_va = FragmentsVa(
        st.session_state.video, st.session_state.predictions_va, st.session_state.hyperparams_va,
        st.session_state.google_cloud)
    st.session_state.trailer_va = TrailerVa(
        st.session_state.video, st.session_state.predictions_va, st.session_state.fragments_va,
        st.session_state.google_cloud)


def on_hyperparam_changed():
    st.session_state.hyperparams_changed = True
    st.session_state.trailer_saved = False


def on_hyperparam_va_changed():
    st.session_state.hyperparams_va_changed = True
    st.session_state.trailer_va_saved = False


def set_hyperparams_base(hyperparams: HyperParamsBase, on_change_func, postfix: str= ''):
    """Sets up hyperparameters for fragments searching."""

    with st.container(border=True):
        for name, title in zip(hyperparams.names, hyperparams.titles):
            if not isinstance(hyperparams.get_possible_options(name), list):
                low_limit, high_limit = hyperparams.get_limits(name)
                init_low_bound, init_high_bound, step = hyperparams.get_init_bounds_and_step(name)
                low_bound, high_bound = hyperparams.get_bounds(name)
                if np.isnan(high_bound):
                    low_bound = st.slider(
                        title, min_value=low_limit, max_value=high_limit,
                        value=init_low_bound, step=step,
                        on_change=on_change_func, key=f'slider_{name}_{postfix}')
                else:
                    low_bound, high_bound = st.slider(
                        title, min_value=low_limit, max_value=high_limit,
                        value=(init_low_bound, init_high_bound), step=step,
                        on_change=on_change_func, key=f'slider_{name}_{postfix}')
                hyperparams.set_bounds(name, low_bound, high_bound)
            else:
                possible_options = hyperparams.get_possible_options(name)
                init_options = hyperparams.get_init_options(name)
                options = st.multiselect(
                    title,
                    options=possible_options,
                    default=init_options,
                    on_change=on_change_func,
                    key=f'multiselect_{name}_{postfix}'
                )
                hyperparams.set_options(name, options)


def set_hyperparams():
    """Sets up hyperparameters for fragments searching."""
    set_hyperparams_base(st.session_state.hyperparams, on_hyperparam_changed)


def set_hyperparams_va():
    """Sets up hyperparameters for fragments searching."""
    set_hyperparams_base(st.session_state.hyperparams_va, on_hyperparam_va_changed, 'va')


def find_fragments_base(changed: str, fragments: FragmentsBase):
    if not st.session_state[changed]:
        return
    with st.spinner('Searching fragments...'):
        fragments.find_fragments()


def find_fragments():
    find_fragments_base('hyperparams_changed', st.session_state.fragments)


def find_fragments_va():
    find_fragments_base('hyperparams_va_changed', st.session_state.fragments_va)


def plot_chart_base(hyperparams_changed_key: str, fig_key: str, chart_key: str, create_func, title: str):

    with st.spinner('Plotting chart...'):
        if st.session_state[hyperparams_changed_key]:
            st.session_state[fig_key] = create_func()
        with st.expander(title):
            st.plotly_chart(st.session_state[fig_key], use_container_width=True, key=chart_key)


def plot_chart():
    plot_chart_base(
        'hyperparams_changed',
        'chart_figure',
        'chart', create_chart,
        'Emotion-Probability chart'
    )


def plot_chart_va():
    plot_chart_base(
        'hyperparams_va_changed',
        'chart_va_figure',
        'chart_va', create_chart_va,
        'Valence-Arousal chart'
    )


def create_chart():

    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )
    fig.add_scatter(
        x=st.session_state.predictions.data['time'],
        y=st.session_state.predictions.data['emotion'],
        name='Emotion',
        line_width=1, line_shape='hv',
        line_color=DEFAULT_COLORS[0],
        opacity=0.5,
        secondary_y=False,
        legendgroup='unfiltered_prediction',
        legendgrouptitle_text="Raw prediction",
        visible='legendonly',
    )
    fig.add_scatter(
        x=st.session_state.predictions.data['time'],
        y=st.session_state.predictions.data['probability'],
        name='Probability',
        line_width=1, line_shape='spline',
        line_color=DEFAULT_COLORS[1],
        opacity=0.5,
        secondary_y=True,
        legendgroup='unfiltered_prediction',
        visible='legendonly',
    )
    fig.add_scatter(
        x=st.session_state.predictions.data['time'],
        y=st.session_state.predictions.data['filtered_emotion'],
        name='Emotion',
        line_width=2, line_shape='hv',
        line_color=DEFAULT_COLORS[0],
        secondary_y=False,
        legendgroup='prediction',
        legendgrouptitle_text="Prediction",
    )
    fig.add_scatter(
        x=st.session_state.predictions.data['time'],
        y=st.session_state.predictions.data['filtered_probability'],
        name='Probability',
        line_width=2, line_shape='spline',
        line_color=DEFAULT_COLORS[1],
        secondary_y=True,
        legendgroup='prediction',
    )
    fig.add_hline(
        y=st.session_state.hyperparams.get_bounds('min_probability')[0],
        line_width=1, line_dash='dash', line_color=DEFAULT_COLORS[3],
        annotation_text='Min. probability', annotation_position="top right",
        secondary_y=True,
    )
    for fragment, (emotion, start_step, end_step, start, peak, end,
                   start_probability, peak_probability, end_probability
                   ) in st.session_state.fragments.data[
        ['emotion', 'start_step', 'end_step', 'start', 'peak', 'end',
         'start_probability', 'peak_probability', 'end_probability']
    ].iterrows():
        color_index = 8 if fragment % 2 == 0 else 2
        fig.add_vline(
            x=peak,
            line_width=1, line_dash='dot',
            line_color=DEFAULT_COLORS[color_index]
        )
        fig.add_vrect(
            x0=start,
            x1=end,
            line_width=0,
            fillcolor=DEFAULT_COLORS[color_index], opacity=0.10,
            annotation_text=f'#{fragment+1}', annotation_position="bottom left",
        )
    fig.update_xaxes(
        title='Time',
    )
    fig.update_yaxes(
        title='Emotion', secondary_y=False
    )
    fig.update_yaxes(
        title='Probability', secondary_y=True
    )
    fig.update_layout(
        xaxis_tickformat='%M:%S',
        margin_t=30, margin_b=0,
        legend_orientation="h",
        legend_y=-0.3
    )
    return fig


def create_chart_va():

    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )
    fig.add_scatter(
        x=st.session_state.predictions_va.data['time'],
        y=st.session_state.predictions_va.data['valence'],
        name='Valence',
        line_width=1, line_shape='spline',
        line_color=DEFAULT_COLORS[0],
        opacity=0.5,
        secondary_y=False,
        legendgroup='unfiltered_prediction',
        legendgrouptitle_text="Raw prediction",
        visible='legendonly',
    )
    fig.add_scatter(
        x=st.session_state.predictions_va.data['time'],
        y=st.session_state.predictions_va.data['arousal'],
        name='Arousal',
        line_width=1, line_shape='spline',
        line_color=DEFAULT_COLORS[1],
        opacity=0.5,
        secondary_y=True,
        legendgroup='unfiltered_prediction',
        visible='legendonly',
    )
    fig.add_scatter(
        x=st.session_state.predictions_va.data['time'],
        y=st.session_state.predictions_va.data['filtered_valence'],
        name='Valence',
        line_width=2, line_shape='spline',
        line_color=DEFAULT_COLORS[0],
        secondary_y=False,
        legendgroup='prediction',
        legendgrouptitle_text="Prediction",
    )
    fig.add_scatter(
        x=st.session_state.predictions_va.data['time'],
        y=st.session_state.predictions_va.data['filtered_arousal'],
        name='Arousal',
        line_width=2, line_shape='spline',
        line_color=DEFAULT_COLORS[1],
        secondary_y=True,
        legendgroup='prediction',
    )
    fig.add_hline(
        y=st.session_state.hyperparams_va.get_bounds('emotion_valence')[0],
        line_width=1, line_dash='dash', line_color=DEFAULT_COLORS[0],
        annotation_text='Min. valence', annotation_position="top left",
        secondary_y=False,
    )
    fig.add_hline(
        y=st.session_state.hyperparams_va.get_bounds('emotion_valence')[1],
        line_width=1, line_dash='dash', line_color=DEFAULT_COLORS[0],
        annotation_text='Max. valence', annotation_position="top left",
        secondary_y=False,
    )
    fig.add_hline(
        y=st.session_state.hyperparams_va.get_bounds('emotion_arousal')[0],
        line_width=1, line_dash='dash', line_color=DEFAULT_COLORS[1],
        annotation_text='Min. arousal', annotation_position="top right",
        secondary_y=True,
    )
    fig.add_hline(
        y=st.session_state.hyperparams_va.get_bounds('emotion_arousal')[1],
        line_width=1, line_dash='dash', line_color=DEFAULT_COLORS[1],
        annotation_text = 'Max. arousal', annotation_position = "top right",
        secondary_y = True,
    )
    for fragment, (start_step, end_step, start, peak, end,
                   start_arousal, peak_arousal, end_arousal
                   ) in st.session_state.fragments_va.data[
        ['start_step', 'end_step', 'start', 'peak', 'end',
         'start_arousal', 'peak_arousal', 'end_arousal']
    ].iterrows():
        color_index = 8 if fragment % 2 == 0 else 2
        fig.add_vline(
            x=peak,
            line_width=1, line_dash='dot',
            line_color=DEFAULT_COLORS[color_index]
        )
        fig.add_vrect(
            x0=start,
            x1=end,
            line_width=0,
            fillcolor=DEFAULT_COLORS[color_index], opacity=0.10,
            annotation_text=f'#{fragment+1}', annotation_position="bottom left",
        )
    fig.update_xaxes(
        title='Time',
    )
    fig.update_yaxes(
        title='Valence', secondary_y=False
    )
    fig.update_yaxes(
        title='Arousal', secondary_y=True
    )
    fig.update_layout(
        xaxis_tickformat='%M:%S',
        margin_t=30, margin_b=0,
        legend_orientation="h",
        legend_y=-0.3
    )

    return fig


def on_fragments_select():
    st.session_state.trailer_saved = False


def on_fragments_va_select():
    st.session_state.trailer_va_saved = False


def display_fragments_table_base(
        hyperparams_changed_key: str, table_data_key: str, table_config_key: str, create_func,
        trailer: TrailerBase, on_fragments_select_func) -> [int]:

    with st.spinner('Displaying table...'):
        if st.session_state[hyperparams_changed_key]:
            st.session_state[table_data_key], st.session_state[table_config_key] = create_func()
        if st.session_state[table_data_key].empty:
            st.warning('No fragments found')
            selected_rows = []
        else:
            event = st.dataframe(st.session_state[table_data_key], use_container_width=True,
                                 column_config=st.session_state[table_config_key],
                                 on_select=on_fragments_select_func,
                                 selection_mode='multi-row', hide_index=False)
            selected_rows = event.selection.rows
            total_time = st.session_state[table_data_key]['time'].iloc[selected_rows].sum()
            st.markdown(f'Total time: {total_time:.2f}s')

        trailer.select_fragments(selected_rows)


def display_fragments_table():
    display_fragments_table_base(
        'hyperparams_changed',
        'table_data',
        'table_config',
        create_fragments_table,
        st.session_state.trailer,
        on_fragments_select
    )


def display_fragments_va_table():
    display_fragments_table_base(
        'hyperparams_va_changed',
        'table_va_data',
        'table_va_config',
        create_fragments_va_table,
        st.session_state.trailer_va,
        on_fragments_va_select
    )


def create_fragments_table() -> [int]:

    df = pd.DataFrame(columns=['emotion', 'start', 'peak', 'end', 'time',
                               'peak_probability', 'screenshot_url', 'probability_chart'])
    df[['emotion', 'start', 'peak', 'end', 'time', 'peak_probability']] = st.session_state.fragments.data[
        ['emotion', 'start', 'peak', 'end', 'time', 'peak_probability']]
    df['screenshot_url'] = st.session_state.trailer.data['screenshot_url']
    df['time'] = st.session_state.fragments.data['time'].dt.total_seconds()

    for fragment, (start_step, end_step) in st.session_state.fragments.data[['start_step', 'end_step']].iterrows():
        df.at[fragment, 'probability_chart'] = (
            st.session_state.predictions.data.loc[start_step: end_step, 'filtered_probability'].to_list())
    df.index += 1

    column_config = {
        '_index': st.column_config.NumberColumn('#'),
        'screenshot_url': st.column_config.ImageColumn('Screenshot'),
        'start': st.column_config.TimeColumn('Start', format='m:ss.SSS'),
        'peak': st.column_config.TimeColumn('Peak', format='m:ss.SSS'),
        'end': st.column_config.TimeColumn('End', format='m:ss.SSS'),
        'time': st.column_config.NumberColumn('Time [s]', format='%.2f'),
        'peak_probability': st.column_config.NumberColumn('Peak probability', format='%.3f'),
        'probability_chart': st.column_config.AreaChartColumn(
            'Probability',
            y_min=st.session_state.predictions.data['filtered_probability'].min(),
            y_max=st.session_state.predictions.data['filtered_probability'].max()
        )
    }

    return df, column_config


def create_fragments_va_table() -> [int]:

    df = pd.DataFrame(columns=['start', 'peak', 'end', 'time', 'peak_arousal', 'screenshot_url',
                               'valence_chart', 'arousal_chart'])
    df[['start', 'peak', 'end', 'time', 'peak_arousal']] = st.session_state.fragments_va.data[
        ['start', 'peak', 'end', 'time', 'peak_arousal']]
    df['screenshot_url'] = st.session_state.trailer_va.data['screenshot_url']
    df['time'] = st.session_state.fragments_va.data['time'].dt.total_seconds()

    for fragment, (start_step, end_step) in st.session_state.fragments_va.data[['start_step', 'end_step']].iterrows():
        df.at[fragment, 'valence_chart'] = (
            st.session_state.predictions_va.data.loc[start_step: end_step, 'filtered_valence'].to_list())
        df.at[fragment, 'arousal_chart'] = (
            st.session_state.predictions_va.data.loc[start_step: end_step, 'filtered_arousal'].to_list())
    df.index += 1

    column_config = {
        '_index': st.column_config.NumberColumn('#'),
        'screenshot_url': st.column_config.ImageColumn('Screenshot'),
        'start': st.column_config.TimeColumn('Start', format='m:ss.SSS'),
        'peak': st.column_config.TimeColumn('Peak', format='m:ss.SSS'),
        'end': st.column_config.TimeColumn('End', format='m:ss.SSS'),
        'time': st.column_config.NumberColumn('Time [s]', format='%.2f'),
        'peak_arousal': st.column_config.NumberColumn('Peak arousal', format='%.3f'),
        'valence_chart': st.column_config.AreaChartColumn(
            'Valence',
            y_min=st.session_state.predictions_va.data['filtered_valence'].min(),
            y_max=st.session_state.predictions_va.data['filtered_valence'].max()
        ),
        'arousal_chart': st.column_config.AreaChartColumn(
            'Arousal',
            y_min=st.session_state.predictions_va.data['filtered_arousal'].min(),
            y_max=st.session_state.predictions_va.data['filtered_arousal'].max()
        )
    }

    return df, column_config


def save_and_download_trailer_base(
        trailer: TrailerBase,
        trailer_saved_key: str,
        save_button_key: str,
        download_button_key: str
):
    save = st.button('Create trailer...',
                     icon=':material/movie:',
                     disabled=len(trailer.selected_fragments) == 0 or st.session_state[trailer_saved_key],
                     key=save_button_key)
    if save:
        with st.spinner('Creating trailer...'):
            st.session_state[trailer_saved_key] = True
            trailer.save()
            st.rerun()

    if st.session_state[trailer_saved_key]:
        st.video(trailer.trailer_file_path)
        with open(trailer.trailer_file_path, mode='rb') as file:
            st.download_button(
                'Download the trailer...',
                file,
                trailer.trailer_file_path,
                'video' + Path(trailer.trailer_file_path).suffix[1:],
                icon=':material/download:', key=download_button_key
            )


def save_and_download_trailer():
    save_and_download_trailer_base(
        st.session_state.trailer,
        'trailer_saved',
        'trailer_save_button',
        'trailer_download_button',
    )


def save_and_download_trailer_va():
    save_and_download_trailer_base(
        st.session_state.trailer_va,
        'trailer_va_saved',
        'trailer_va_save_button',
        'trailer_va_download_button'
    )


def show_title_and_help():
    st.markdown('<h2 style="text-align: center;">Trailer creator</h2>', unsafe_allow_html=True)
    with st.expander('See brief help...', icon=':material/help:'):
        st.markdown(get_help())


def main():
    st.markdown(
        "# Video"
    )

    c1, _ = st.columns(2, gap='medium')
    with c1:
        st.markdown('##  One person video')
        upload_video()
        if st.session_state.uploaded_file is None:
            st.info('Please upload a video first.', icon=":material/info:")
            return

    c1, _ = st.columns(2, gap='medium')
    with c1:
        st.video(st.session_state.uploaded_file)
        create_common_components()

    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.markdown('### 1st type')
    with c2:
        st.markdown('### 2nd type')
    with c1:
        create_components()
    with c2:
        create_components_va()
    with c1:
        set_hyperparams()
        find_fragments()
    with c2:
        set_hyperparams_va()
        find_fragments_va()
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        plot_chart()
    with c2:
        plot_chart_va()
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.session_state.trailer.create_data()
        display_fragments_table()
        if st.session_state.trailer.data.empty:
            st.error('No fragment found. Please change settings.', icon=":material/error:")
        elif len(st.session_state.trailer.selected_fragments) == 0:
            st.info('Please select fragments for trailer.', icon=":material/info:")
        st.session_state.hyperparams_changed = False
    with c2:
        st.session_state.trailer_va.create_data()
        display_fragments_va_table()
        if st.session_state.trailer_va.data.empty:
            st.error('No fragment found. Please change settings.', icon=":material/error:")
        elif len(st.session_state.trailer_va.selected_fragments) == 0:
            st.info('Please select fragments for trailer.', icon=":material/info:")
        st.session_state.hyperparams_va_changed = False

    c1, _ = st.columns(2, gap='medium')
    with c1:
        st.markdown('## Trailer')
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.markdown('### 1st type')
    with c2:
        st.markdown('### 2nd type')
    with c1:
        save_and_download_trailer()
    with c2:
        save_and_download_trailer_va()


main()