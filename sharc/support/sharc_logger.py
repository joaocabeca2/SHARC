# -*- coding: utf-8 -*-
import os
import sys
import yaml
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logging:
    @staticmethod
    def setup_logging(
        default_path='support/logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG',
    ):
        """
        Setup logging configuration.
        """
        path = os.getenv(env_key, default_path)
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)


class SimulationLogger:
    """
    Logs simulation metadata to a YAML file for reproducibility.
    """

    def __init__(self, param_file: str, log_base: str = "simulation_log"):
        """
        Initialize the logger with the parameter file path.

        Args:
            param_file (str): Path to the simulation parameter file.
            log_base (str): Subdirectory for storing logs (default: 'simulation_log').
        """
        self.param_file = Path(param_file).resolve()
        self.param_name = self.param_file.stem
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.output_dir = (
            self.param_file.parent.parent
            / "output"
            / log_base
            / f"simulation_{self.param_name}_{self.timestamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / f"simulation_log_{self.timestamp}.yaml"
        self.start_time = None
        self.root_dir = self._get_root_dir()

        self.data = {
            "repo": self._get_git_info(),
            "root_dir": str(self.root_dir) if self.root_dir else "N/A",
            "run": {
                "command": self._get_invocation_command(),
                "python_version": self._get_python_version(),
                "pkgs": self._get_installed_packages(),
            },
        }

    def start(self):
        """
        Start the simulation timer and record initial metadata.
        """
        self.start_time = datetime.now()
        self.data["run"]["started_at"] = self.start_time.isoformat()

    def end(self):
        """
        Stop the simulation timer, compute duration, and save the YAML log.
        """
        end_time = datetime.now()
        self.data["run"]["ended_at"] = end_time.isoformat()
        if self.start_time:
            duration = end_time - self.start_time
            self.data["run"]["duration"] = str(duration)
        with open(self.log_path, "w") as f:
            yaml.dump(self.data, f, sort_keys=False, allow_unicode=True)

    def _get_root_dir(self, folder_name: str = "sharc") -> Optional[Path]:
        path = self.param_file.resolve()
        for parent in path.parents:
            if (parent / folder_name).exists():
                return parent
        return None

    def _run_git_cmd(self, args: list[str]) -> Optional[str]:
        try:
            return subprocess.check_output(
                ['git'] + args, stderr=subprocess.DEVNULL
            ).decode().strip()
        except subprocess.CalledProcessError:
            return None

    def _get_git_info(self) -> dict:
        branch = self._run_git_cmd(['rev-parse', '--abbrev-ref', 'HEAD'])
        commit = self._run_git_cmd(['rev-parse', 'HEAD'])
        remote_name = (
            self._run_git_cmd(['config', f'branch.{branch}.remote'])
            if branch
            else None
        )
        remote_url = (
            self._run_git_cmd(['config', f'remote.{remote_name}.url'])
            if remote_name
            else None
        )
        return {
            "url": remote_url or "N/A",
            "branch": branch or "N/A",
            "commit": commit or "N/A",
        }

    def _get_invocation_command(self) -> str:
        return f"{sys.executable} {' '.join(sys.argv)}"

    def _get_python_version(self) -> str:
        return sys.version.replace("\n", " ")

    def _get_installed_packages(self) -> list[str]:
        try:
            pkgs = subprocess.check_output(
                [sys.executable, '-m', 'pip', 'freeze'],
                stderr=subprocess.DEVNULL,
            )
            return sorted(pkgs.decode().strip().split('\n'))
        except subprocess.CalledProcessError:
            return ["Could not retrieve packages"]
