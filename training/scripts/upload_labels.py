"""Upload YOLO polygon labels to Label Studio as pre-annotations."""

from pathlib import Path

import typer

from digitex.config.settings import get_settings
from digitex.label_studio.uploader import LabelStudioUploader


def main(
    project_id: int = typer.Option(..., help="Label Studio project ID"),
    batch_size: int = typer.Option(50, help="Tasks per import batch"),
) -> None:
    """Upload YOLO polygon labels to Label Studio as pre-annotations."""
    settings = get_settings()
    ls = settings.label_studio

    if not ls.api_key:
        raise typer.BadParameter("Set LABEL_STUDIO_API_KEY in .env")

    labels_dir = (
        settings.paths.training_dir
        / settings.data.data_dir_name
        / settings.data.data_type_dir_name
        / settings.data.raw_data_dir_name
        / "labels"
    )

    uploader = LabelStudioUploader(ls)
    uploader.upload(
        project_id=project_id,
        labels_dir=labels_dir,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    typer.run(main)
