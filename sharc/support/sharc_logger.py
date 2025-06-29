# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 18:15:47 2017

@author: edgar
"""

import os
import logging.config
import yaml
import subprocess
import sys

from datetime import datetime
from pathlib import Path


class Logging():

    @staticmethod
    def setup_logging(
        default_path='support/logging.yaml',
        default_level=logging.INFO, env_key='LOG_CFG',
    ):
        """
        Setup logging configuration
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)


class SimulationLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / f"simulation_output_log_{str(output_dir)}.yaml"
        self.start_time = None
        self.data = {
            "repo": self._get_git_info(),
            "run": {
                "command": self._get_invocation_command(),
                "python_version": self._get_python_version(),
                "pkgs": self._get_installed_packages(),
            }
        }

    def _run_git_cmd(self, args):
        try:
            return subprocess.check_output(['git'] + args, stderr=subprocess.DEVNULL).decode().strip()
        except subprocess.CalledProcessError:
            return None

    def _get_git_info(self):
        branch = self._run_git_cmd(['rev-parse', '--abbrev-ref', 'HEAD'])
        commit = self._run_git_cmd(['rev-parse', 'HEAD'])
        remote_name = self._run_git_cmd(['config', f'branch.{branch}.remote']) if branch else None
        remote_url = self._run_git_cmd(['config', f'remote.{remote_name}.url']) if remote_name else None
        return {
            "url": remote_url or "N/A",
            "branch": branch or "N/A",
            "commit": commit or "N/A"
        }

    def _get_invocation_command(self):
        return f"{sys.executable} {' '.join(sys.argv)}"

    def _get_python_version(self):
        return sys.version.replace("\n", " ")

    def _get_installed_packages(self):
        try:
            pkgs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.DEVNULL)
            return sorted(pkgs.decode().strip().split('\n'))
        except subprocess.CalledProcessError:
            return ["Could not retrieve packages"]

    def start(self):
        self.start_time = datetime.now()
        self.data["run"]["started_at"] = self.start_time.isoformat()

    def end(self):
        end_time = datetime.now()
        self.data["run"]["ended_at"] = end_time.isoformat()
        if self.start_time:
            duration = end_time - self.start_time
            self.data["run"]["duration"] = str(duration)

        with open(self.log_path, "w") as f:
            yaml.dump(self.data, f, sort_keys=False)
