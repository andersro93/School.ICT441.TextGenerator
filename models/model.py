from nt import DirEntry
import os

import self as self
from keras import Model as KerasModel


class Model(object):
    """
    Base class for the different models in this project
    """

    _files: dict = {}
    """ The files that is used to parse the model """

    _raw_content: str = ''
    """ The files contents as one long string """

    _assets_path: str = 'assets_test'
    """ Path to the assets to read from """

    _model: KerasModel = None
    """ The Keras model that is used """

    _model_path: str = None
    """ The path where to save and use the model from """

    _activation_method: str = 'softmax'
    """ Activation method to use """

    _optimizer: str = 'adam'
    """ Optimizer method to use """

    _loss_method: str = 'categorical_crossentropy'
    """ Loss method to use """

    _training_data_encoding: str = 'utf-8'
    """ The encoding used on the training data """

    def print_model_summary(self) -> self:
        """
        Returns a model summary if the model has been created
        :return: str
        """
        if self._model:
            self._model.summary()
            return

        print('No model has been created yet')

        return self

    def load_weights(self, weights: str) -> self:
        """
        Loads weights from the given weights file
        :param weights:
        :return: self
        """
        if not self._model:
            print('No model has been created, please create the model first!')
            return

        self._model.load_weights(weights)
        self.compile_model()

        return self

    def compile_model(self) -> self:
        """
        Compiles the model and runs some optimizer on it
        :return: self
        """
        if not self._model:
            print('No model has been created, please create the model first!')
            return

        self._model.compile(loss=self._loss_method, optimizer=self._optimizer)

        return self

    def _read_data_from_assets(self) -> self:
        """
        Reads and parses the data from the assets folder into the object itself
        :return: self
        """
        for directory in os.scandir(self._get_assets_full_path()):
            self._parse_directory(directory)

        return self

    def _concat_assets_content_to_one_string(self) -> self:
        """
        Concatenates the contents from all the assets to one string
        :return: self
        """
        for key, value in self._files.items():
            self._raw_content = self._raw_content + value

        self._raw_content = self._raw_content.lower()

        return self

    def _parse_directory(self, directory: DirEntry) -> self:
        """
        Recursively parses the given directory and starts to parse any found files
        :param directory:
        :return: self
        """
        entry: DirEntry
        for entry in os.scandir(directory):
            if entry.is_dir():
                self._parse_directory(entry)
            else:
                self._parse_file(entry)

        return self

    def _parse_file(self, file: DirEntry) -> self:
        """
        Tries to parse the given file and puts it in self._files dictionary
        :param file: DirEntry
        :return: self
        """
        data: str
        try:
            with open(file.path, 'r', encoding=self._training_data_encoding) as file_reader:
                data = file_reader.read()
                file_reader.close()

        except Exception:
            print(f"Unable to parse file: {file.path}")
            return

        self._files[file.path] = data

        return self

    def _get_assets_full_path(self) -> str:
        """
        Returns a computed full path to the directory where the assets are located as a string
        :return: str
        """
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), self._assets_path)
