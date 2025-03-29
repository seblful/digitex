import tkinter as tk
from tkinter import ttk


class ExtractorApp:
    def __init__(self, root) -> None:
        self.root = root
        self.setup_root()

        self.ui = UserInterface(self.root)

    def setup_root(self) -> None:
        self.root.title("Testing Digitizer")
        self.root.state('zoomed')

    def run(self) -> None:
        self.root.mainloop()


class UserInterface:
    def __init__(self, root) -> None:
        self.root = root
        self.setup_ui()

    def setup_ui(self) -> None:
        self.setup_menubar()

    def setup_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        # file_menu.add_command(label="Open", command=self.open_pdf)
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.root.config(menu=menubar)
