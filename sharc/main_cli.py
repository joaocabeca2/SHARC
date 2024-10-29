# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:17:14 2017

@author: edgar
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sharc.support.logging import Logging
from sharc.controller import Controller
from sharc.gui.view_cli import ViewCli
from sharc.model import Model

import getopt



def main(argv):
    print("Welcome to SHARC!\n")

    param_file = ''

    try:
        opts, args = getopt.getopt(argv, "hp:")
    except getopt.GetoptError:
        print("usage: main_cli.py -p <param_file>")
        sys.exit(2)

    if not opts:
        param_file = os.path.join(os.getcwd(), "input", "parameters_wifi.yaml")
    else:
        for opt, arg in opts:
            if opt == "-h":
                print("usage: main_cli.py -p <param_file>")
                sys.exit()
            elif opt == "-p":
                param_file = param_file = os.path.join(os.getcwd(), arg)

    Logging.setup_logging()

    model = Model()
    view_cli = ViewCli()
    controller = Controller()

    view_cli.set_controller(controller)
    controller.set_model(model)
    model.add_observer(view_cli)

    view_cli.initialize(param_file)


if __name__ == "__main__":
    main(sys.argv[1:])
