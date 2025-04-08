import itertools
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from auxiliary import css_styling, GoogleCloud
from zipfile import ZipFile
from pathlib import Path


EMOTIONS = {
    'anger': (-0.41, 0.79), # anger, rage
    'contempt': (-0.57, 0.66), # contempt
    'disgust': (-0.67, 0.49), # disgust
    'fear': (-0.12, 0.78), # fear
    'happy': (0.9, 0.16), # happiness
    'neutral': (0.0, 0.0), # neutral
    'sad': (-0.82, -0.4), # sadness
    'surprise': (0.37, 0.91), # surprise
    'uncertain': (-0.5, 0.0), # uncertain
}

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

MODEL_ON_TOP_DENSE_NUMS = [1, 2] # Options for the number of additional fully connected layers

MODEL_ON_TOP_DENSE_UNITS = [512, 1024] # Options for the number of output neurons in the additional fully connected layer

MODEL_ON_TOP_DROPOUT_RATES = [.0, .2] # Options for the proportion of data to drop before feeding into the fully connected layer during training


def load_data():

    st.session_state.google_cloud = GoogleCloud()

    # emotions
    st.session_state.emotions = EMOTIONS

    # base models
    @st.cache_data(show_spinner=False)
    def _get_base_models():
        df = pd.DataFrame(
            [(name, size, inference_time) for name, (size, inference_time) in KERAS_BASE_MODELS.items()],
            columns=['name', 'size', 'accuracy']
        ).set_index('name')
        return df

    st.session_state.base_models = _get_base_models()

    # model on top configs
    @st.cache_data(show_spinner=False)
    def _get_model_on_top_configs():
        dropout_rate_dense_units_combs = [
            (dropout_rate, dense_units) for dense_units in MODEL_ON_TOP_DENSE_UNITS
            for dropout_rate in MODEL_ON_TOP_DROPOUT_RATES
        ]
        model_on_top_configs = []
        for dense_num in MODEL_ON_TOP_DENSE_NUMS:
            model_on_top_configs += set(
                itertools.permutations(dropout_rate_dense_units_combs, dense_num)
            ).union(
                set(
                    itertools.combinations_with_replacement(dropout_rate_dense_units_combs, dense_num)
                )
            )
        df = pd.DataFrame(model_on_top_configs)
        df[1] = df[1].apply(lambda t: (None, None) if t is None else t)
        df[['dropouts_1', 'units_1']] = df[0].to_list()
        df[['dropouts_2', 'units_2']] = df[1].to_list()
        df.drop(columns=[0, 1], inplace=True)
        df.index = df.apply(
            lambda config:
            f'({config[0]}, {config[1]})' + (f', ({config[2]}, {config[3]})' if not np.isnan(config[2]) else ''),
            axis=1
        )
        df = df.convert_dtypes()
        columns = pd.MultiIndex.from_tuples(
            [
                ('Block #1', 'Dropout Rate'),
                ('Block #1', 'Dense Units'),
                ('Block #2', 'Dropout Rate'),
                ('Block #2', 'Dense Units'),
            ],
            names=['block', 'layer']
        )
        df.columns = columns
        return df

    st.session_state.models_on_top =_get_model_on_top_configs()

    # train size
    @st.cache_data(show_spinner=False)
    def _load_train_size():
        file = st.session_state.google_cloud.open_file(
            file_path='classification/image_preprocessing/train_face_extraction.csv',
            mode='rt'
        )
        df = pd.read_csv(file, index_col='emotion')
        df['count'] = df['failed_images'] + df['detected_faces']
        df['percent'] = df['count'] / sum(df['count']) * 100
        df.drop(columns=['failed_images', 'detected_faces'], inplace=True)
        return df

    st.session_state.train_size = _load_train_size()

    # base model sizes
    @st.cache_data(show_spinner=False)
    def _load_base_model_sizes():
        file = st.session_state.google_cloud.open_file(
            file_path='classification/keras_base_models_processing/base_model_sizes.csv',
            mode='rt'
        )
        df = pd.read_csv(
            file,
            usecols=['base_model_name', 'image_size', 'feature_size'],
            index_col='base_model_name'
        ).sort_index()
        return df

    st.session_state.base_model_sizes = _load_base_model_sizes()

    # base models processing params
    @st.cache_data(show_spinner=False)
    def _load_base_models_processing_params():
        file = st.session_state.google_cloud.open_file(
            file_path='classification/keras_base_models_processing/pipeline_base_models_processing.csv',
            mode='rt'
        )
        df = pd.read_csv(file)
        return df['params'].apply(eval)

    st.session_state.base_models_processing = _load_base_models_processing_params()

    @st.cache_data(show_spinner=False)
    def _load_base_model_inference_times():
        # inference times
        file = st.session_state.google_cloud.open_file(
            file_path='classification/keras_base_models_processing/model_inference_times.csv',
            mode='rt'
        )
        df_1 = pd.read_csv(
            file,
            usecols=['base_model_name', 'inference_time'],
            index_col='base_model_name'
        )
        df_1['inference_time'] *= 1000
        file = st.session_state.google_cloud.open_file(
            file_path='va/keras_base_models_processing/model_inference_times.csv',
            mode='rt'
        )
        df_2 = pd.read_csv(
            file,
            usecols=['base_model_name', 'inference_time'],
            index_col='base_model_name'
        )
        df_2['inference_time'] *= 1000
        df = pd.merge(df_1, df_2, on='base_model_name')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.base_model_inference_times = _load_base_model_inference_times()

    @st.cache_data(show_spinner=False)
    def _load_base_model_scores():
        file = st.session_state.google_cloud.open_file(
            file_path='classification/keras_base_models_processing/base_model_selection.csv',
            mode='rt'
        )
        df_1 = pd.read_csv(
            file,
            usecols=['base_model_name', 'top1_accuracy_score', 'inference_time_score', 'weighted_score', 'rank'],
            index_col=['base_model_name', 'top1_accuracy_score']
        )
        file = st.session_state.google_cloud.open_file(
            file_path='va/keras_base_models_processing/base_model_selection.csv',
            mode='rt'
        )
        df_2 = pd.read_csv(
            file,
            usecols=['base_model_name', 'top1_accuracy_score', 'inference_time_score', 'weighted_score', 'rank'],
            index_col=['base_model_name', 'top1_accuracy_score']
        )
        df = pd.merge(df_1, df_2, on=['base_model_name', 'top1_accuracy_score']).sort_index()
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.base_model_scores = _load_base_model_scores()

    # train face extraction
    @st.cache_data(show_spinner=False)
    def _load_train_face_extraction():
        file = st.session_state.google_cloud.open_file('classification/image_preprocessing/train_face_extraction.csv')
        df_1 = pd.read_csv(
            file,
            index_col='emotion')
        df_1['percent'] = df_1['failed_images'] / (df_1['detected_faces'] + df_1['failed_images']) * 100
        file = st.session_state.google_cloud.open_file('va/image_preprocessing/train_face_extraction.csv')
        df_2 = pd.read_csv(
            file,
            index_col='emotion')
        df_2['percent'] = df_2['failed_images'] / (df_2['detected_faces'] + df_2['failed_images']) * 100
        df = pd.merge(df_1, df_2, on='emotion')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.train_face_extraction = _load_train_face_extraction()

    @st.cache_data(show_spinner=False)
    def _load_train_extraction_faces():
        df_1 = pd.DataFrame(columns=['emotion', 'extracted_url', 'not_extracted_url']).set_index('emotion')
        for emotion in st.session_state.emotions:
            urls = st.session_state.google_cloud.get_blob_urls(
                f'classification/presentation/extracted_faces/{emotion}/*.jpg'
            )
            urls += st.session_state.google_cloud.get_blob_urls(
                f'classification/presentation/not_extracted_faces/{emotion}/*.jpg'
            )
            df_1.loc[emotion] = urls
        df_2 = pd.DataFrame(columns=['emotion', 'extracted_url', 'not_extracted_url']).set_index('emotion')
        for emotion in st.session_state.emotions:
            urls = st.session_state.google_cloud.get_blob_urls(
                f'va/presentation/extracted_faces/{emotion}/*.jpg'
            )
            urls += st.session_state.google_cloud.get_blob_urls(
                f'va/presentation/not_extracted_faces/{emotion}/*.jpg'
            )
            df_2.loc[emotion] = urls
        df = pd.merge(df_1, df_2, on='emotion')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.train_faces = _load_train_extraction_faces()

    # train face cleaning
    @st.cache_data(show_spinner=False)
    def _load_train_face_cleaning():
        file = st.session_state.google_cloud.open_file('classification/image_preprocessing/train_cleaning.csv')
        df_1 = pd.read_csv(file, index_col='emotion')
        total_number = df_1['failed_number'] + df_1['remain_number']
        df_1['duplicated_percent'] = df_1['duplicated_number'] / total_number * 100
        df_1['different_percent'] = df_1['different_number'] / total_number * 100
        df_1['failed_percent'] = df_1['failed_number'] / total_number * 100
        df_1['remain_percent'] = df_1['remain_number'] / total_number * 100
        file = st.session_state.google_cloud.open_file('va/image_preprocessing/train_cleaning.csv')
        df_2 = pd.read_csv(file, index_col='emotion')
        total_number = df_2['failed_number'] + df_2['remain_number']
        df_2['duplicated_percent'] = df_2['duplicated_number'] / total_number * 100
        df_2['different_percent'] = df_2['different_number'] / total_number * 100
        df_2['failed_percent'] = df_2['failed_number'] / total_number * 100
        df_2['remain_percent'] = df_2['remain_number'] / total_number * 100
        df = pd.merge(df_1, df_2, on='emotion')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.train_face_cleaning = _load_train_face_cleaning()

    @st.cache_data(show_spinner=False)
    def _load_train_duplicated_faces():
        df_1 = pd.DataFrame(columns=['emotion', 'original_url', 'duplicate_url']).set_index('emotion')
        for emotion in st.session_state.emotions:
            urls = st.session_state.google_cloud.get_blob_urls(
                f'classification/presentation/duplicated_images/{emotion}/*.jpg'
            )
            df_1.loc[emotion] = urls
        df_2 = pd.DataFrame(columns=['emotion', 'original_url', 'duplicate_url']).set_index('emotion')
        for emotion in st.session_state.emotions:
            urls = st.session_state.google_cloud.get_blob_urls(
                f'va/presentation/duplicated_images/{emotion}/*.jpg'
            )
            df_2.loc[emotion] = urls
        df = pd.merge(df_1, df_2, on='emotion')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.train_duplicated_faces = _load_train_duplicated_faces()

    @st.cache_data(show_spinner=False)
    def _load_train_different_faces():
        df_1 = pd.DataFrame(columns=['emotion', 'url']).set_index('emotion')
        for emotion in st.session_state.emotions:
            urls = st.session_state.google_cloud.get_blob_urls(
                f'classification/presentation/different_images/{emotion}/*.jpg'
            )
            df_1.loc[emotion] = urls
        df_2 = pd.DataFrame(columns=['emotion', 'url']).set_index('emotion')
        for emotion in st.session_state.emotions:
            urls = st.session_state.google_cloud.get_blob_urls(
                f'va/presentation/different_images/{emotion}/*.jpg'
            )
            df_2.loc[emotion] = urls
        df = pd.merge(df_1, df_2, on='emotion')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.train_different_faces = _load_train_different_faces()

    @st.cache_data(show_spinner=False)
    def _load_similarity():
        similarity_limits = []
        similarity_hists = []
        similarity_medians_limits = []
        similarity_medians_hists = []

        file = st.session_state.google_cloud.open_file(
            'classification/presentation/similarity/similarity.pkl', mode='rb'
        )
        limits, hists, medians_limits, medians_hists = pickle.load(file)
        file.close()
        similarity_limits.append(limits)
        similarity_hists.append(hists)
        similarity_medians_limits.append(medians_limits)
        similarity_medians_hists.append(medians_hists)

        file= st.session_state.google_cloud.open_file(
            'va/presentation/similarity/similarity.pkl', mode='rb'
        )
        limits, hists, medians_limits, medians_hists = pickle.load(file)
        file.close()
        similarity_limits.append(limits)
        similarity_hists.append(hists)
        similarity_medians_limits.append(medians_limits)
        similarity_medians_hists.append(medians_hists)

        return similarity_limits, similarity_hists, similarity_medians_limits, similarity_medians_hists

    (st.session_state.similarity_limits,
     st.session_state.similarity_hists,
     st.session_state.similarity_medians_limits,
     st.session_state.similarity_medians_hists) = _load_similarity()

    # model on top selection
    @st.cache_data(show_spinner=False)
    def _load_model_on_top_selection():
        file = st.session_state.google_cloud.open_file('classification/model_building/model_on_top_selection.csv')
        df_1 = pd.read_csv(
            file, index_col='model_on_top_config', usecols=['model_on_top_config', 'best_epoch', 'score'])
        file.close()
        file = st.session_state.google_cloud.open_file('va/model_building/model_on_top_selection.csv')
        df_2 = pd.read_csv(
            file, index_col='model_on_top_config', usecols=['model_on_top_config', 'best_epoch', 'score'])
        file.close()
        df = pd.merge(df_1, df_2, on='model_on_top_config')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.model_on_top_selection = _load_model_on_top_selection()

    @st.cache_data(show_spinner=False)
    def _load_model_on_top_selection_logs():
        file = st.session_state.google_cloud.open_file(
            'classification/presentation/model_on_top_selection/model_on_top_selection_logs.csv'
        )
        df_1 = pd.read_csv(file, index_col=['model', 'epoch'])
        file.close()
        file = st.session_state.google_cloud.open_file(
            'va/presentation/model_on_top_selection/model_on_top_selection_logs.csv'
        )
        df_2 = pd.read_csv(file, index_col=['model', 'epoch'])
        file.close()
        df = pd.merge(df_1, df_2, on=['model', 'epoch'])
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.model_on_top_selection_logs = _load_model_on_top_selection_logs()

    @st.cache_data(show_spinner=False)
    def _load_model_fine_tuning_logs():
        file = st.session_state.google_cloud.open_file(
            'classification/presentation/model_fine_tuning/model_fine_tuning_logs.csv'
        )
        df_1 = pd.read_csv(file, index_col='epoch')
        file.close()
        file = st.session_state.google_cloud.open_file(
            'va/presentation/model_fine_tuning/model_fine_tuning_logs.csv'
        )
        df_2 = pd.read_csv(file, index_col='epoch')
        file.close()
        df = pd.merge(df_1, df_2, on='epoch')
        columns = pd.MultiIndex.from_tuples(
            [(1, column) for column in df_1.columns] + [(2, column) for column in df_2.columns],
            names=['type', 'param']
        )
        df.columns = columns
        return df

    st.session_state.model_fine_tuning_logs = _load_model_fine_tuning_logs()

st.set_page_config(page_title='Human emotions recognition', page_icon=':smile:', layout='wide')

css_styling()

with st.spinner('Loading data...'):
    load_data()

st.session_state.pages = {
    '': [
        st.Page('sections/title.py', title="Title", default=True),
        st.Page('sections/intro.py', title='Intro'),
        st.Page('sections/provided_data.py', title='Provided Data'),
        st.Page('sections/models_architecture.py', title='Models architecture'),
    ],
    'Pipelines': [
        st.Page('sections/base_model_selection.py', title='Base model selection'),
        st.Page('sections/image_preprocessing.py', title='Image preprocessing'),
        st.Page('sections/model_building.py', title='Model building'),
    ],
    'Demo': [
        st.Page('sections/camera.py', title="Camera"),
        st.Page('sections/video.py', title="Video"),
    ],
}

pg = st.navigation(st.session_state.pages)
pg.run()
