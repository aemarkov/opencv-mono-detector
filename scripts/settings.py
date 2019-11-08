#
# Helpers to store and load application settings to the YAML file
#

import os
import yaml

class Settings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def load(filename, default=None):
        """
        Load settings from the YAML file
        Args:
            filename Filename of the file to load settings
            default  Default settings object to return if file not found
        Returns:
            New Settings instance
        """
        if default is not None:
           if filename == None or not os.path.isfile(filename):
               return default

        try:
            with open(filename, 'r') as file:
                data =  yaml.load(file.read())
                return Settings.from_dict(data)
        except Exception as ex:
            print('Failed to load config {}: {}'.format(filename, str(ex)))
            return None

    def store(self, filename):
        """
        Store settings to the YAML file
        Args:
            filename Filename of the file to store settings
        """
        data = self.to_dict()
        with open(filename, 'w') as file:
            file.write(yaml.dump(data))

    @staticmethod
    def from_dict(dictionary):
        """
        Crteate new Settings object from dictionary
        Args:
            dictionary Dictionary
        Returns:
            New Settings object
        """
        obj = Settings()
        for key in dictionary:
            if type(dictionary[key]) is dict:
                obj.__dict__[key] = Settings.from_dict(dictionary[key])
            else:
                obj.__dict__[key] = dictionary[key]
        return obj

    def to_dict(self):
        dictionary = {}
        for key in self.__dict__:
            if type(self.__dict__[key]) is Settings:
                dictionary[key] = self.__dict__[key].to_dict()
            else:
                dictionary[key] = self.__dict__[key]
        return dictionary
