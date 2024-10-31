import yaml
from dataclasses import dataclass

@dataclass
class ParametersBase:
    """Base class for parameter dataclassess
    """
    section_name: str = "DEFAULT"
    is_space_to_earth: bool = False # whether the system is a space station or not

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
        
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            

        if not self.section_name in config.keys():
            print(f"ParameterBase: section {self.section_name} not in parameter file.\
                  Only default parameters where loaded.")
            return

        # Load all the parameters from the configuration file
        attr_list = [a for a in dir(self) if not a.startswith('__') and not
                     callable(getattr(self, a)) and a != "section_name"]

        for attr in attr_list:
            try:
                attr_val = getattr(self, attr)
                if isinstance(attr_val, str):
                    setattr(self, attr, config[self.section_name][attr])
                elif isinstance(attr_val, bool):
                    setattr(self, attr, bool(config[self.section_name][attr]))
                elif isinstance(attr_val, float):
                    setattr(self, attr, float(config[self.section_name][attr]))
                elif isinstance(attr_val, int):
                    setattr(self, attr, int(config[self.section_name][attr]))
                elif isinstance(attr_val, tuple):
                    # Check if the string defines a list of floats
                    try:
                        param_val = config[self.section_name][attr]
                        tmp_val = list(map(float, param_val.split(",")))
                        setattr(self, attr, tuple(tmp_val))
                    except ValueError:
                        # its a regular string. Let the specific class implementation
                        # do the sanity check
                        print(f"ParametersBase: could not convert string to tuple \"{self.section_name}.{attr}\"")
                        exit()

            except KeyError:
                print(f"ParametersBase: NOTICE! Configuration parameter \"{self.section_name}.{attr}\" is not set in configuration file. Using default value {attr_val}")
            except Exception as e:
                print(f"Um erro desconhecido foi encontrado: {e}")