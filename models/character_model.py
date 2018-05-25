from typing import Dict, Any

import numpy
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.utils import np_utils

from .model import Model


class CharacterModel(Model):
    """
    Simple model for predicting single characters
    """

    data_x: numpy.ndarray = None
    """ The train data """

    data_y: numpy.ndarray = None
    """ The output data """

    pattern_length: int = 100
    """ The total length of each training pattern """

    __char_to_int_dictionary: Dict[str, int] = None
    """ The dictionary containing the character with their integer value """

    __unique_chars: set = None
    """ A set off all the unique chars """

    def __init__(self):
        """
        CharacterModel constructor
        """
        super(CharacterModel, self).__init__()

        self._model_path = "./generated_models/character-{epoch:02d}-{loss:.4f}.hdf5"

    def prepare_model(self, print_info=False):
        """
        Prepares the model, parses data etc..
        :type print_info: bool, Prints information about parsing if true
        :return: None
        """
        self._read_data_from_assets()
        self._concat_assets_content_to_one_string()

        self.__unique_chars = sorted(list(set(self._raw_content)))
        self.__char_to_int_dictionary = dict((character, index) for index, character in enumerate(self.__unique_chars))

        data_x, data_y = self._format_data_to_x_and_y()

        amount_of_patterns = len(data_x)

        # Assign the data to class
        self.data_x = numpy.reshape(data_x, (amount_of_patterns, self.pattern_length, 1))
        self.data_y = np_utils.to_categorical(data_y)

        if print_info:
            print(f"Total length: {len(self._raw_content)}")
            print(f"Characters: {len(self.__unique_chars)}")
            print(f"Total Patterns: {amount_of_patterns}")

    def generate_text_of_length(self, length: int, verbose=False) -> str:
        """
        Predicts and prints from the given phrase.
        The amount of characters to predict is decided by the parameter length
        :param length: The amount of characters to predict
        :param verbose: Boolean value for verbose generation or not
        :return: str
        """
        # String to return
        return_string = ''

        # Pick a random pattern to use for prediction
        pattern_id = numpy.random.randint(0, len(self.data_x) - 1)

        # Get the selected pattern
        pattern = self.data_x[pattern_id]

        if verbose:
            print('Seed used to print:')
            print(''.join(self.__get_letter_from_number(value) for value in pattern))

        # Predict the given times
        for i in range(length):
            # Format the current phrase into correct vector
            x_input = numpy.reshape(pattern, (1, len(pattern), 1))

            # Get a prediction
            predictions = self._model.predict(x_input)

            # Obtain the predicted letter
            predicted_int_for_letter = numpy.random.choice(len(predictions[0]), p=predictions[0])

            # Append the translated letter to the return string
            return_string += self.__get_letter_from_number(predicted_int_for_letter)

            # Add the letter to the next prediction
            pattern = numpy.append(pattern, predicted_int_for_letter)
            pattern = pattern[1:len(pattern)]

        return return_string

    def create_model(self) -> None:
        """
        Creates the model and preserves it on the model object
        :return: None
        """
        model = Sequential()

        model.add(LSTM(256, input_shape=(self.data_x.shape[1], self.data_x.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))

        model.add(Bidirectional(LSTM(256)))
        model.add(Dropout(0.2))

        model.add(Dense(self.data_y.shape[1], activation='softmax'))

        self._model = model

    def train_model(self, epochs=30, batch_size=150) -> None:
        """
        Starts training the model
        :param epochs: int
        :param batch_size: int
        :return: None
        """

        self.compile_model()

        checkpoint = ModelCheckpoint(self._model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        self._model.fit(
            x=self.data_x,
            y=self.data_y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list
        )

    def __get_letter_from_number(self, number: numpy.ndarray) -> str:
        """
        Helper method for finding the letter for a corresponding number
        :param number:
        :return: str:
        """
        for character in self.__char_to_int_dictionary:
            if self.__char_to_int_dictionary[character] == number:
                return character

        return f'Could not find letter for number: {number}'

    def _format_data_to_x_and_y(self) -> tuple:
        """
        Formats the data to an x and y dictionary
        :return: tuple:
        """
        data_x = []
        data_y = []
        for i in range(0, len(self._raw_content) - self.pattern_length, 1):

            seq_x = self._raw_content[i:i + self.pattern_length]
            seq_y = self._raw_content[i + self.pattern_length]

            data_x.append([self.__char_to_int_dictionary[char] for char in seq_x])
            data_y.append(self.__char_to_int_dictionary[seq_y])

        return data_x, data_y
