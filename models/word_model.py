from keras import Sequential
from keras.layers import LSTM, Dense

from .model import Model


class WordModel(Model):
    """
    Word model that is used to predict whole next words
    """

    def __init__(self):
        """
        Word model constructor
        """
        super(WordModel, self).__init__()

        self._model_path = "./generated_models/word-{epoch:02d}-{loss:.4f}.hdf5"

    def create_model(self) -> None:
        """
        Creates the model and preserves it on the model object
        :return: None
        """
        model = Sequential()

        # First layer
        model.add(
            LSTM(32, activation='relu')
        )

        # Second layer
        model.add(
            Dense(32, activation='relu')
        )

        self._model = model
