import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog

from modules.handlers import PDFHandler
from modules.processors import FileProcessor


class ExtractorApp:
    def __init__(self, root) -> None:
        self.root = root
        self.setup_root()

        # Dirs
        self.inputs_dir = "inputs"
        self.ckpt_path = os.path.join(
            self.inputs_dir, "checkpoints.json")

        self.current_pdf_path = None
        self.current_pdf_obj = None
        self.current_image = None
        self.current_page = 0

        self.ui = UserInterface(self.root, self)

        self.pdf_handler = PDFHandler()
        self.file_processor = FileProcessor()

    def setup_root(self) -> None:
        self.root.title("Testing Digitizer")
        self.root.state('zoomed')

    def run(self) -> None:
        self.root.mainloop()


class UserInterface:
    def __init__(self, root: tk.Tk, main_app: ExtractorApp) -> None:
        self.root = root
        self.main_app = main_app

        self.setup_ui()

    def setup_ui(self) -> None:
        self.setup_menubar()
        self.setup_panes()
        self.setup_status_bar()

    def setup_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.main_app.open_pdf)
        file_menu.add_command(label="Load checkpoints",
                              command=self.main_app.load_ckpt)
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.root.config(menu=menubar)

    def setup_panes(self) -> None:
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.left_pane = ttk.PanedWindow(main_pane)
        self.right_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)

        self.setup_left_frame()

    def setup_left_frame(self) -> None:
        self.left_frame = ttk.Frame(self.left_pane)
        self.left_canvas = tk.Canvas(self.left_frame, bg='white')
        self.left_canvas.pack(expand=True, fill=tk.BOTH)
        # self.setup_navigation_controls(self.left_pane)

    def setup_status_bar(self) -> None:
        self.main_app.status = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN)
        self.main_app.status.pack(side=tk.BOTTOM, fill=tk.X)
