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
        self.root.maxsize(self.root.winfo_screenwidth(),
                          self.root.winfo_screenheight())
        self.root.minsize(int(self.root.winfo_screenwidth() / 2),
                          int(self.root.winfo_screenheight() / 2))

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

        self.left_width_weight = 3
        self.right_width_weight = 7
        self.right_weights = [1, 1]

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
        # Main horizontal paned window
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(expand=True, fill=tk.BOTH)

        self.setup_left_pane()
        self.setup_right_pane()

    def setup_left_pane(self) -> None:
        left_frame = ttk.Frame(self.main_pane)
        self.left_canvas = tk.Canvas(left_frame, bg='lightgray')
        self.left_canvas.pack(expand=True, fill=tk.BOTH)
        self.main_pane.add(left_frame, weight=self.left_width_weight)
        # self.setup_navigation_controls(self.left_pane)

    def setup_right_pane(self) -> None:
        right_frame = ttk.Frame(self.main_pane)
        right_pane = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_pane.pack(expand=True, fill=tk.BOTH)

        # Top frame
        top_frame = ttk.Frame(right_pane)
        top_canvas = tk.Canvas(top_frame, bg='lightgray')
        top_canvas.pack(expand=True, fill=tk.BOTH)
        right_pane.add(top_frame, weight=self.right_weights[0])

        # Bottom frame
        bottom_frame = ttk.Frame(right_pane)
        bottom_canvas = tk.Canvas(bottom_frame, bg="lightgray")
        bottom_canvas.pack(expand=True, fill=tk.BOTH)
        right_pane.add(bottom_frame, weight=self.right_weights[1])

        # Add the vertical paned window to the parent frame
        self.main_pane.add(right_frame, weight=self.right_width_weight)

    def setup_status_bar(self) -> None:
        self.main_app.status = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN)
        self.main_app.status.pack(side=tk.BOTTOM, fill=tk.X)
