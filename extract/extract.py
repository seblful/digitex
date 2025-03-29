import tkinter as tk

from components.extractor import ExtractorApp


def main() -> None:
    root = tk.Tk()
    app = ExtractorApp(root)
    app.run()


if __name__ == "__main__":
    main()
