import streamlit as st
from streamlit_option_menu import option_menu
from auxiliary import remove_blank_space, set_text_style


def intro():
    title = set_text_style('<b>Thesis on computer vision.</b><br>', tag='span', font_size=80) + \
            set_text_style('<br>', tag='span', font_size=20) + \
            set_text_style('Human emotions recognition', tag='span', font_size=64)
    title = set_text_style(title, color="Blue")
    st.markdown(set_text_style('&nbsp;', font_size=36), unsafe_allow_html=True)
    st.markdown(title, unsafe_allow_html=True)
    st.markdown(set_text_style('&nbsp;', font_size=36), unsafe_allow_html=True)
    author = set_text_style('Author: Alexey Sherstobitov</br>', font_size=48, )
    st.markdown(author, unsafe_allow_html=True)


def concept():
    st.markdown('<h1 style="text-align:center; color:blue">Model types</h1>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.markdown('<h2 style="text-align:center">1st type</h2>', unsafe_allow_html=True)
        st.markdown('---')
        st.markdown('### Predicts the probability of human emotion based on facial expression')
    with c2:
        st.markdown('<h2 style="text-align:center">2nd type</h2>', unsafe_allow_html=True)
        st.markdown('---')
        st.markdown('### Predicts levels of valence and intensity (arousal) of human emotion based on facial expression')
    c1, c2 = st.columns(2, gap='medium')
    st.markdown('<h3 style="text-align:center; color:darkorange">Learning</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200283993-4ec70b6c-9da7-4355-891a-564332559041.svg',
                 use_container_width=True)
        st.markdown('')
    with c2:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200284152-5faabe63-43ab-4593-828b-2684ef35b70f.svg',
                 use_container_width=True)
        st.markdown('')
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.markdown('''
    ```python
    EMOTIONS = (
        'anger', # anger, rage
        'contempt', # contempt
        'disgust', # disgust
        'fear', # fear
        'happy', # happiness
        'neutral', # neutral
        'sad', # sadness
        'surprise', # surprise
        'uncertain', # uncertain
    )
    ```                ''')
    with c2:
        st.markdown('''
    ```python
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
    ```                        ''')
    st.markdown('<h3 style="text-align:center; color:darkorange">Inference</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200292114-4c6a4e79-a151-463c-a3d9-e22bfa526efa.svg',
                 use_container_width=True)
        st.markdown('')
    with c2:
        st.image('https://user-images.githubusercontent.com/107345313/'
                 '200292181-72ec54e2-f367-4c06-854f-ce541722d108.svg',
                 use_container_width=True)
        st.markdown('')


def learning():
    st.markdown('<h1 style="text-align:center; color:blue">Model building</h1>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('<h2 style="text-align:center">Pipeline</h2>', unsafe_allow_html=True)
    c2.markdown('<h2 style="text-align:center">Stage</h2>', unsafe_allow_html=True)
    c3.markdown('<h2 style="text-align:center">Tasks</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('---')
    c1.markdown('#### 1. Collecting information about base models in [Keras Applications]'
                '(https://keras.io/api/applications/)')
    c2.markdown('---')
    c2.markdown('#### 1.1. Collecting information about sizes of input images and '
                'feature vectors')
    c3.markdown('---')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Collecting information about the optimal input image sizes for each base model</span></li>
    <li><span style="fontSize:24px">Collecting information about the sizes of feature vectors at the output of the last pooling layer of each "base model"</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 1.2. Inference time measuring')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Estimation of the inference time of a model consisting of a base model combined with the "heaviest" upper model</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 1.3. Base model selection')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Selecting a base model with the best combination of performance and accuracy, shown on the ImageNet validation dataset</li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('---')
    c1.markdown('#### 2. Image preprocessing')
    c2.markdown('---')
    c2.markdown('#### 2.1. Extracting face images from the train dataset')
    c3.markdown('---')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Cropping images to the face area</span></li>
    <li><span style="fontSize:24px">Exclude images that do not contain faces or that contain only partial faces
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 2.2. Extracting face images from the test dataset')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Cropping images to the face area</span></li>
    <li><span style="fontSize:24px">Exclude images that do not contain faces or that contain only partial faces
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 2.3. Cleaning up the training dataset')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Excluding identical or very similar face images</span></li>
    <li><span style="fontSize:24px">Excluding incorrectly labeled images
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c1.markdown('---')
    c1.markdown('#### 3. Model building')
    c2.markdown('---')
    c2.markdown('#### 3.1. Feature extraction from the train dataset')
    c3.markdown('---')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Collecting a dataset of feature vectors from the train dataset faces for use in selecting the best model on top</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.2. Feature extraction from the test dataset')
    text = '''
    <ul>
   <li><span style="fontSize:24px">Collecting a dataset of feature vectors from the test dataset faces for use in selecting the best model on top</span></li>
     </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.3. Selecting the best model on top')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Finding the top model configuration that has the best potential for use in the resulting model</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.4. Model on top learning')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Reducing the likelihood of too large a gradient occurring 
    when training an entire model, which could lead to the destruction of the underlying model</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.5. Model fine tuning')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Improving the model's prediction accuracy</span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)
    _, c2, c3 = st.columns([25, 30, 45], gap='medium')
    c2.markdown('#### 3.6. Model testing')
    text = '''
    <ul>
    <li><span style="fontSize:24px">Testing is performed on an image from the built-in camera of the local computer
    </span></li>
    </ul>
    '''
    c3.markdown(text, unsafe_allow_html=True)


def results():
    st.markdown('<h1 style="text-align:center; color:blue">Model selection</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown('<h2 style="text-align:center">Base model</h3>', unsafe_allow_html=True)
    c1.markdown("#### :orange[1st type]: EfficientNetB0, input=224, feature size=1280")
    c1.markdown("#### :orange[2nd type]: EfficientNetB0, input=224, feature size=1280")
    c2.markdown('<h2 style="text-align:center">On top model</h3>', unsafe_allow_html=True)
    c2.markdown("#### :orange[1st type]: (dropout=0.0, units=512), (dropout=0.0, units=256)")
    c2.markdown("#### :orange[2nd type]: (dropout=0.0, units=2048), (dropout=0.0, units=2048)")
    st.markdown("---")
    st.markdown('<h1 style="text-align:center; color:blue">Model learning</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown('<h2 style="text-align:center">On top model learning</h3>', unsafe_allow_html=True)
    c1.markdown("#### :orange[1st type]: epoch=1, score=0.3402")
    c1.markdown("#### :orange[2nd type]: epoch=1, score=0.1658")
    c2.markdown('<h2 style="text-align:center">Model fine tuning</h3>', unsafe_allow_html=True)
    c2.markdown("#### :orange[1st type]: epoch=22, score=0.519395")
    c2.markdown("#### :orange[2nd type]: epoch=31, score=0.3306")
    st.markdown('---')
    st.markdown('<h1 style="text-align:center; color:blue">Model testing</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<h2 style="text-align:center">In process</h3>', unsafe_allow_html=True)
    c1.image('https://user-images.githubusercontent.com/107345313/'
             '204358144-a0d4f282-84d9-4879-9b47-1fa3f4709131.png')
    c2.markdown('<h2 style="text-align:center">At the end</h3>', unsafe_allow_html=True)
    c2.image('https://user-images.githubusercontent.com/107345313/'
             '204358140-37ff0cf8-7f70-4a75-966b-1aa2aaf9300c.png')
    c3.markdown('<h2 style="text-align:center">Results</h3>', unsafe_allow_html=True)
    c3.image('https://user-images.githubusercontent.com/107345313/'
             '204360810-2e86e4c3-e5bd-4fbe-8dbf-5d9ecf614c8c.png')


st.set_page_config(page_title='Human emotions recognition', page_icon=':smile:', layout='wide')

with st.sidebar:
    choice = option_menu(
        '',
        options=[
            "Title",
            "Introduction",
            "Process",
            "---",
            "Outcomes",
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
    case "Title":
        intro()
    case "Introduction":
        concept()
    case "Process":
        learning()
    case 'Outcomes':
        results()
