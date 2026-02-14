from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import inspect
import os
from typing import Dict, Type

from .. import __version__


class BaseCLI(ABC):
    """Base class for all CLI tools."""

    prog: str = ""
    description: str = ""
    backend_arg_name: str = ""
    backend_short_flag: str = ""
    default_backend_env: str = ""
    default_backend: str = ""
    backend_module = None
    backend_base_class: Type = None

    def _available_backends(self) -> Dict[str, Type]:
        module = self.backend_module
        base_cls = self.backend_base_class
        if module is None or base_cls is None:
            return {}
        return {
            obj.name: obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, base_cls) and not inspect.isabstract(obj)
        }

    def _build_backend_parser(self, choices: list[str]) -> argparse.ArgumentParser:
        default = os.environ.get(self.default_backend_env, self.default_backend)
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            self.backend_short_flag,
            f"--{self.backend_arg_name}",
            choices=choices,
            default=default,
            help=f"Backend to use (default: {default})",
        )
        return parser

    @abstractmethod
    def _add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add tool-specific arguments."""

    def _add_backend_args(
        self,
        parser: argparse.ArgumentParser,
        backend_args: argparse.Namespace,
        available: Dict[str, Type],
    ) -> None:
        """Add backend-specific CLI arguments."""
        backend_name = getattr(backend_args, self.backend_arg_name)
        available[backend_name].contribute_to_cli(parser)

    def build_parser(
        self, available: Dict[str, Type], args=None
    ) -> argparse.ArgumentParser:
        backend_parser = self._build_backend_parser(choices=sorted(available.keys()))
        backend_args, _ = backend_parser.parse_known_args(args)

        parser = argparse.ArgumentParser(
            prog=self.prog,
            description=self.description,
            parents=[backend_parser],
        )
        self._add_arguments(parser)
        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
        )
        self._add_backend_args(parser, backend_args, available)
        return parser

    @abstractmethod
    def _run(self, args: argparse.Namespace, available: Dict[str, Type]) -> int:
        """Execute the tool logic."""

    def run(self) -> int:
        """Entry point: discover backends, parse args, run."""
        available = self._available_backends()
        parser = self.build_parser(available)
        args = parser.parse_args()
        return self._run(args, available)
