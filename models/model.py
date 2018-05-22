from nt import DirEntry
import os


class Model(object):
    """
    Base class for the different models in this project
    """

    _files = {}
    """ The files that is used to parse the model """

    _raw_content = ''
    """ The files contents as one long string """

    _assets_path = 'assets_test'
    """ Path to the assets to read from """

    _model = None
    """ The Keras model that is used """

    _model_path = None
    """ The path where to save and use the model from """

    def print_model_summary(self) -> None:
        """
        Returns a model summary if the model has been created
        :return: str
        """
        if self._model:
            self._model.summary()
            return

        print('No model has been created yet')

    def load_weights(self, weights: str) -> None:
        """
        Loads weights from the given weights file
        :param weights:
        :return: None
        """
        if not self._model:
            print('No model has been created, please create the model first!')
            return

        self._model.load_weights(weights)
        self.compile_model()

    def compile_model(self) -> None:
        """
        Compiles the model and runs some optimizer on it
        :return: None
        """
        if not self._model:
            print('No model has been created, please create the model first!')
            return

        self._model.compile(loss='categorical_crossentropy', optimizer='adam')

    def _read_data_from_assets(self) -> None:
        """
        Reads and parses the data from the assets folder into the object itself
        :return: None
        """
        for directory in os.scandir(self._get_assets_full_path()):
            self._parse_directory(directory)

    def _concat_assets_content_to_one_string(self) -> None:
        """
        Concatenates the contents from all the assets to one string
        :return: None
        """
        for key, value in self._files.items():
            self._raw_content = self._raw_content + value

        self._raw_content = self._raw_content.lower()

    def _parse_directory(self, directory: DirEntry) -> None:
        """
        Recursively parses the given directory and starts to parse any found files
        :param directory:
        :return: None
        """
        entry: DirEntry
        for entry in os.scandir(directory):
            if entry.is_dir():
                self._parse_directory(entry)
            else:
                self._parse_file(entry)

    def _parse_file(self, file: DirEntry) -> None:
        """
        Tries to parse the given file and puts it in self._files dictionary
        :param file: DirEntry
        :return: None
        """
        data: str
        try:
            with open(file.path, 'r', encoding="utf-8") as file_reader:
                data = file_reader.read()
                file_reader.close()

        except Exception:
            print(f"Unable to parse file: {file.path}")
            return

        self._files[file.path] = data

    def _get_assets_full_path(self) -> str:
        """
        Returns a computed full path to the directory where the assets are located as a string
        :return: str
        """
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), self._assets_path)
