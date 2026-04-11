"""File processing utilities."""

import json
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger()


class FileProcessor:
    """Processor for file I/O operations supporting various formats."""

    @staticmethod
    def read_txt(txt_path: str | Path) -> list[str]:
        """Read lines from a text file.

        Args:
            txt_path: Path to the text file.

        Returns:
            List of lines from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
        """
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            content = txt_file.readlines()

        return content

    @staticmethod
    def write_txt(txt_path: str | Path, lines: list[str]) -> None:
        """Write lines to a text file.

        Args:
            txt_path: Path to the output text file.
            lines: List of lines to write.

        Raises:
            IOError: If the file cannot be written.
        """
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.writelines(lines)

    @staticmethod
    def read_json(json_path: str | Path) -> dict:
        """Read JSON data from a file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            Dictionary containing the JSON data.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def write_json(
        json_dict: dict,
        json_path: str | Path,
        indent: int = 4,
    ) -> None:
        """Write data to a JSON file.

        Args:
            json_dict: Dictionary to write as JSON.
            json_path: Path to the output JSON file.
            indent: Number of spaces for indentation.

        Raises:
            IOError: If the file cannot be written.
            TypeError: If the data is not JSON serializable.
        """
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_dict, json_file, indent=indent, ensure_ascii=False)

    @staticmethod
    def write_yaml(
        yaml_path: str | Path,
        data: dict,
        comment: str | None = None,
    ) -> None:
        """Write data to a YAML file.

        Args:
            yaml_path: Path to the output YAML file.
            data: Dictionary to write as YAML.
            comment: Optional comment to add at the top of the file.

        Raises:
            IOError: If the file cannot be written.
        """
        with open(yaml_path, "w", encoding="utf-8") as yaml_file:
            if comment:
                yaml_file.write(comment)
            yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)
