import streamlit as st
from streamlit_option_menu import option_menu
from auxiliary import remove_blank_space, set_text_style


def intro():
    title = set_text_style('<b>Дипломная работа по компьютерному зрению.</b><br>', tag='span', font_size=80) + \
            set_text_style('<br>', tag='span', font_size=20) + \
            set_text_style('Распознавание эмоций человека', tag='span', font_size=64)
    title = set_text_style(title, color="Blue")
    st.markdown(set_text_style('&nbsp;', font_size=36), unsafe_allow_html=True)
    st.markdown(title, unsafe_allow_html=True)
    st.markdown(set_text_style('&nbsp;', font_size=36), unsafe_allow_html=True)
    author = set_text_style('Автор: Алексей Шерстобитов</br>', font_size=48, )
    st.markdown(author, unsafe_allow_html=True)


def concept():
    st.markdown('<h1 style="text-align:center; color:blue">Типы моделей</h1>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.markdown('<h2 style="text-align:center">Тип 1</h2>', unsafe_allow_html=True)
        st.markdown('### По выражению лица человека предсказывает вероятности эмоций, которые он испытывает')
    with c2:
        st.markdown('<h2 style="text-align:center">Тип 2</h2>', unsafe_allow_html=True)
        st.markdown('### По выражению лица человека распознает уровни валентности (valence) '
                    'и интенсивности (arousal) эмоции, которую он испытывает')
    c1, c2 = st.columns(2, gap='medium')
    st.markdown('<h3 style="text-align:center; color:darkorange">Обучение</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200283993-4ec70b6c-9da7-4355-891a-564332559041.svg',
                 use_column_width=True)
        st.markdown('')
    with c2:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200284152-5faabe63-43ab-4593-828b-2684ef35b70f.svg',
                 use_column_width=True)
        st.markdown('')
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.markdown('''
    ```python
    EMOTIONS = (
        'anger', # гнев, злость
        'contempt', # презрение
        'disgust', # отвращение
        'fear', # страх
        'happy', # веселый
        'neutral', # нейтральный
        'sad', # грусть
        'surprise', # удивленность
        'uncertain', # неуверенность
    )
    ```
                ''')
    with c2:
        st.markdown('''
    ```python
    EMOTIONS = {
        'anger': (-0.41, 0.79), # гнев, злость
        'contempt': (-0.57, 0.66), # презрение
        'disgust': (-0.67, 0.49), # отвращение
        'fear': (-0.12, 0.78), # страх
        'happy': (0.9, 0.16), # веселый
        'neutral': (0.0, 0.0), # нейтральный
        'sad': (-0.82, -0.4), # грусть
        'surprise': (0.37, 0.91), # удивленность
        'uncertain': (-0.5, 0.0), # неуверенность
    }
    ```                        ''')
    st.markdown('<h3 style="text-align:center; color:darkorange">Инференс</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200292114-4c6a4e79-a151-463c-a3d9-e22bfa526efa.svg',
                 use_column_width=True)
        st.markdown('')
    with c2:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200292181-72ec54e2-f367-4c06-854f-ce541722d108.svg',
                 use_column_width=True)
        st.markdown('')


def learning():
    st.markdown('<h1 style="text-align:center; color:blue">Процесс создания моделей</h1>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('<h2 style="text-align:center">Пайплайн</h2>', unsafe_allow_html=True)
    c2.markdown('<h2 style="text-align:center">Этап</h2>', unsafe_allow_html=True)
    c3.markdown('<h2 style="text-align:center">Задачи</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('---')
    c1.markdown('#### 1. Сбор информации о базовых моделях в [Keras Applications]'
                '(https://keras.io/api/applications/)')
    c2.markdown('---')
    c2.markdown('#### 1.1. Получение информации о размерах входных изображений и '
                'векторов признаков')
    c3.markdown('---')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Получение информации об оптимальных размерах входных изображений для каждой базовой 
    модели</span></li>
    <li><span style="fontSize:24px">Получение информации о размерах векторов признаков на выходе последнего слоя 
    пуллинга каждой "базовой модели"</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 1.2. Измерение времени инференса')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Оценка времени инференса модели, состоящей из базовой модели в сочетании с самой 
    "тяжелой" верхней моделью</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 1.3. Выбор базовой модели')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Выбор базовой модели с наилучшим сочетанием быстродействия и точности модели, 
    показанной на валидационном датасете ImageNet</li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('---')
    c1.markdown('#### 2. Предобработка изображений')
    c2.markdown('---')
    c2.markdown('#### 2.1. Извлечение изображений лиц из тренировочного датасета')
    c3.markdown('---')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Обрезка изображений до области лица</span></li>
    <li><span style="fontSize:24px">Исключение изображений, на которых нет изображений лиц, либо они показаны частично
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 2.2. Извлечение изображений лиц из тестового датасета')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Обрезка изображений до области лица</span></li>
    <li><span style="fontSize:24px">Исключение изображений, на которых нет изображений лиц, либо они показаны частично
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 2.3. Очистка тренировочного датасета')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Исключение одинаковых или очень похожих изображений лиц</span></li>
    <li><span style="fontSize:24px">Исключение ошибочно размеченных изображений
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('---')
    c1.markdown('#### 3. Создание модели')
    c2.markdown('---')
    c2.markdown('#### 3.1. Извлечение признаков из тренировочного датасета')
    c3.markdown('---')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Получение датасета из векторов признаков изображений лиц тренировочного датасета для 
    использования при выборе лучшей полносвязной модели</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.2. Извлечение признаков из тестового датасета')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Получение датасета из векторов признаков изображений лиц тестового датасета для 
    использования при выборе лучшей полносвязной модели</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.3. Выбор лучшей верхней модели')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Определение конфигурации верхней модели, имеющей наилучший потенциал для применения 
    в составе результирующей модели</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.4. Обучение полносвязной модели')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Снижение вероятности возникновения слишком большой величины градиента при обучении 
    модели целиком, которое может привести к разрушению базовой модели</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.5. Тонкая настройка модели')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Улучшение точности предсказаний модели в целом</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.6. Тестирование работы модели')
    text = '''
    <ul>
    <li><span style="fontSize:24px">ОТестирование производится на изображении со встроенной камеры локального компьютера
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)


def results():
    st.markdown('<h2 style="text-align:center; color:blue">Модель</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown('<h3 style="text-align:center">Базовая модель</h3>', unsafe_allow_html=True)
    c1.markdown("##### :orange[Тип 1]: EfficientNetB0, input=224, feature size=1280")
    c1.markdown("##### :orange[Тип 2]: EfficientNetB0, input=224, feature size=1280")
    c2.markdown('<h3 style="text-align:center">Верхняя модель</h3>', unsafe_allow_html=True)
    c2.markdown("##### :orange[Тип 1]: (dropout=0.0, units=512), (dropout=0.0, units=256)")
    c2.markdown("##### :orange[Тип 2]: (dropout=0.0, units=2048), (dropout=0.0, units=2048)")
    st.markdown("---")
    st.markdown('<h2 style="text-align:center; color:blue">Обучение</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown('<h3 style="text-align:center">Обучение верхней модели</h3>', unsafe_allow_html=True)
    c1.markdown("##### :orange[Тип 1]: epoch=1, score=0.3402")
    c1.markdown("##### :orange[Тип 2]: epoch=1, score=0.1658")
    c2.markdown('<h3 style="text-align:center">Тонкая настройка модели</h3>', unsafe_allow_html=True)
    c2.markdown("##### :orange[Тип 1]: epoch=22, score=0.519395")
    c2.markdown("##### :orange[Тип 2]: epoch=31, score=0.3306")
    st.markdown('---')
    st.markdown('<h2 style="text-align:center; color:blue">Тестирование</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<h3 style="text-align:center">В процессе</h3>', unsafe_allow_html=True)
    c1.image('https://user-images.githubusercontent.com/107345313/'
             '204358144-a0d4f282-84d9-4879-9b47-1fa3f4709131.png')
    c2.markdown('<h3 style="text-align:center">По завершении</h3>', unsafe_allow_html=True)
    c2.image('https://user-images.githubusercontent.com/107345313/'
             '204358140-37ff0cf8-7f70-4a75-966b-1aa2aaf9300c.png')
    c3.markdown('<h3 style="text-align:center">Результат</h3>', unsafe_allow_html=True)
    c3.image('https://user-images.githubusercontent.com/107345313/'
             '204360810-2e86e4c3-e5bd-4fbe-8dbf-5d9ecf614c8c.png')


st.set_page_config(page_title='Распознавание эмоций человека', page_icon=':person:', layout='wide')

with st.sidebar:
    choice = option_menu(
        '',
        options=[
            "Титул",
            "Введение",
            "Процесс",
            "---",
            "Результаты",
        ],
        icons=[
            "",
            "map",
            "gear",
            '',
            'book',
            '',
        ],
        orientation='vertical',
        key='main_menu'
    )

match choice:
    case "Титул":
        intro()
    case "Введение":
        concept()
    case "Процесс":
        learning()
    case 'Результаты':
        results()
