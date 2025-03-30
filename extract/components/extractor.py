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

    def open_pdf(self) -> None:
        pdf_path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf")])
        if not pdf_path:
            return

        self.load_pdf(pdf_path)
        self.save_ckpt()
        self.update_status(f"Opened PDF file: {pdf_path}")

    def load_pdf(self,
                 pdf_path: str,
                 current_page: int = 0) -> None:
        self.current_pdf_path = pdf_path
        self.current_pdf_obj = self.pdf_handler.open_pdf(pdf_path)
        self.current_page = current_page

        # Show image
        pdf_page = self.current_pdf_obj[self.current_page]
        self.current_image = self.pdf_handler.get_page_image(pdf_page)
        self.show_image()

    def show_image(self) -> None:
        if self.current_image:
            self.original_image = ImageTk.PhotoImage(self.current_image)
            self.canvas_image = self.ui.left_canvas.create_image(
                0, 0, anchor=tk.NW, image=self.original_image)
            self.ui.left_canvas.config(
                scrollregion=self.ui.left_canvas.bbox(tk.ALL))

    def load_ckpt(self) -> None:
        ckpt = self.file_processor.read_json(self.ckpt_path)

        if not ckpt:
            self.update_status(
                f"Failed loading checkpoint")
            return

        self.load_pdf(pdf_path=ckpt["pdf_path"], current_page=ckpt["page"])
        self.update_status(
            f"Opened PDF file {self.current_pdf_path} from checkpoint")

    def save_ckpt(self) -> None:
        ckpt = {"pdf_path": self.current_pdf_path,
                "page": self.current_page}

        self.file_processor.write_json(ckpt, self.ckpt_path)

    def update_status(self, message) -> None:
        self.status.config(text=message)
        self.root.update_idletasks()

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
