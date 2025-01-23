from collections import namedtuple
from PIL import Image, ImageOps
import numpy as np
from keras import models
from typing import Union

EMOTIONS = {
    'anger': (-0.41, 0.79), # anger, rage
    'contempt': (-0.57, 0.66), # contempt
    'disgust': (-0.67, 0.49), # disgust
    'fear': (-0.12, 0.78), # fear
    'happy': (0.9, 0.16), # cheerful
    'neutral': (0.0, 0.0), # neutral
    'sad': (-0.82, -0.4), # sadness
    'surprise': (0.37, 0.91), # surprise
    'uncertain': (-0.5, 0.0), # uncertainty
}

EmotionPrediction = namedtuple('EmotionPrediction', ['emotion', 'probability'])
ValenceArousalPrediction = namedtuple('ValenceArousalPrediction', ['emotion', 'error', 'valence', 'arousal'])

class FaceEmotionRecognitionNet:

    def __init__(self, model_path: str, emotions=None):
        """
        file_path: path to the saved model file.
        emotions: predicted emotions.
        """
        # Load the model
        if emotions is None:
            emotions = EMOTIONS
        self.__model = models.load_model(filepath=model_path, compile=False)
        self.__emotions = emotions

    def predict(self, face_image: np.array) -> Union[EmotionPrediction, ValenceArousalPrediction]:
        """Predict a person's emotion based on their face.
        
        Arguments:
        - face_image: image of a person's face.
        """
        image = Image.fromarray(face_image)
        size = max(image.width, image.height)
        # Make the image square
        padded_image = ImageOps.pad(image, (size, size))
        # Adjust the image size
        resized_image = padded_image.resize(self.__model.input_shape[1:3])
        # Get the prediction
        tensor = np.asarray(resized_image)[None, ...]
        predicts = self.__model.predict(tensor, verbose=0)[0]
        # Prepare the resulting data
        if isinstance(self.__emotions, (list, tuple)):
            probability = predicts.max()
            label = predicts.argmax()
            emotion = self.__emotions[label]
            return EmotionPrediction(emotion, probability)
        else:
            valence, arousal = predicts
            dists = np.apply_along_axis(lambda a: np.linalg.norm(a - np.array(list(self.__emotions.values())), axis=1),
                                        arr=predicts[None, ...], axis=1)
            error = dists.min()
            label = dists.argmin()
            emotion = list(self.__emotions.keys())[label]
            return ValenceArousalPrediction(emotion, error, valence, arousal)
