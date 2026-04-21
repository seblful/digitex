"""Training CLI commands."""

import typer

from digitex.logging import setup_logging

setup_logging()

app = typer.Typer(help="Training commands for YOLO model.")


@app.callback()
def callback() -> None:
    """Training commands for YOLO model.
    
    Currently no commands available. Use digitex-extract for extraction commands.
    """
    pass


if __name__ == "__main__":
    app()
