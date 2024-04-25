import configparser
from dataclasses import dataclass

@dataclass
class ParametersBase:
    """Base class for parameter dataclassess
    """
    section_name: str = "DEFAULT"

    def load_parameters_from_file(self, config_file: str):
        """Load the parameters from file.
        The sanity check is child class reponsibility

        Parameters
        ----------
        file_name : str
            the path to the configuration file

        Raises
        ------
        ValueError
            if a parameter is not valid
        """
        config = configparser.ConfigParser()
        config.read(config_file)

        if not self.section_name in config.sections():
            print(f"ParameterBase: section {self.section_name} not in parameter file.\
                  Only default parameters where loaded.")
            return

        # Load all the parameters from the configuration file
        attr_list = [a for a in dir(self) if not a.startswith('__') and not
                     callable(getattr(self, a)) and not "section_name"]
        for param in attr_list:
            if isinstance(param, str):
                setattr(self, param, config.get(self.section_name, param))
            elif isinstance(param, int):
                setattr(self, param, config.getint(self.section_name, param))
            elif isinstance(param, float):
                setattr(self, param, config.getfloat(self.section_name, param))
            elif isinstance(param, bool):
                setattr(self, param, config.getboolean(self.section_name, param))
