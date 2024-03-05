# SHARC
Welcome to SHARC, a simulator for use in SHARing and Compatibility studies of radiocommunication systems. The development of this software is being lead by the Telecommunications Regulatory Authority (TRA) of Brazil, ANATEL, and it implements the framework proposed by Recommendation ITU-R M.2101 for "modelling and simulation of IMT networks and systems for use in sharing and compatibility studies".

# How-to
- Install python 3.12 or above in your system
- Install virtualenv module to create a virtual envoriment
- Go to the `SHARC` source code root diretory
- Create a virutal enviroment called `venv` make sure you are calling the right python version here:
    `python3 -m venv venv`
- Activate the environment:
    `source venv/bin/activate`
- Install the package requirements with pip:
    `pip install -r requirements.txt`
- Move to `sharc` directory.
- Run the default scenario with `python main_cli.py -p input/parameters.ini`
- Check the `output`repository for the simulation results.