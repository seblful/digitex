import os

from PIL import Image, ImageTk
import ctypes

import tkinter as tk
from tkinter import ttk, filedialog

from modules.handlers import PDFHandler, ImageHandler
from modules.processors import FileProcessor


class ExtractorApp:
    def __init__(self, root) -> None:
        self.root = root

        # Instances
        self.ui = UserInterface(self.root, self)
        self.ui.setup_ui()

        self.pdf_handler = PDFHandler()
        self.image_handler = ImageHandler()
        self.file_processor = FileProcessor()

        # Paths
        self.inputs_dir = "inputs"
        self.ckpt_path = os.path.join(
            self.inputs_dir, "checkpoints.json")

        # Current
        self.current_pdf_path = None
        self.current_pdf_obj = None
        self.current_page = 0
        self.page_number = 0
        self.zoom_level = 1.0
        self.base_image_width = 595
        self.base_image_height = 842

        # Images
        self.original_image = None
        self.base_image = None
        self.resized_image = None
        self.tk_image = None
        self.canvas_image = None

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
        self.page_number = len(self.current_pdf_obj)

        # Show image
        pdf_page = self.current_pdf_obj[self.current_page]
        self.original_image = self.pdf_handler.get_page_image(pdf_page)
        self.base_image = self.image_handler.resize_image(
            self.original_image, self.base_image_width, self.base_image_height)
        self.zoom_level = 1.0  # Reset zoom when loading new PDF
        self._resize_and_display_image()

    def _resize_and_display_image(self) -> None:
        if not self.base_image:
            return

        canvas_width = self.ui.left_canvas.winfo_width()
        canvas_height = self.ui.left_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Apply zoom or fit to canvas
        if self.zoom_level == 1.0:
            # Fit to canvas
            self.resized_image = self.image_handler.resize_image(
                self.base_image, canvas_width, canvas_height)
        else:
            # Apply zoom level
            width = int(self.base_image.width * self.zoom_level)
            height = int(self.base_image.height * self.zoom_level)
            self.resized_image = self.base_image.resize(
                (width, height), Image.Resampling.LANCZOS)

        # Update the image on canvas
        self.tk_image = ImageTk.PhotoImage(self.resized_image)
        if self.canvas_image:
            self.ui.left_canvas.itemconfig(
                self.canvas_image, image=self.tk_image)
        else:
            self.canvas_image = self.ui.left_canvas.create_image(
                0, 0, anchor=tk.NW, image=self.tk_image)

        self.ui.left_canvas.config(
            scrollregion=self.ui.left_canvas.bbox(tk.ALL))

    def on_mousewheel(self, event: tk.Event) -> None:
        # Check if Control key is pressed (state 0x0004 is Control on Windows)
        if event.state & 0x0004:
            # Zoom in/out
            if event.delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            # Scroll vertically
            scroll_amount = -event.delta // 120  # 1 step per wheel click
            self.ui.left_canvas.yview_scroll(scroll_amount, "units")
        return 'break'  # Prevent other handlers

    def zoom_in(self) -> None:
        self.zoom_level *= 1.1
        self._resize_and_display_image()

    def zoom_out(self) -> None:
        self.zoom_level /= 1.1
        self._resize_and_display_image()

    def load_ckpt(self) -> None:
        ckpt = self.file_processor.read_json(self.ckpt_path)

        if not ckpt:
            self.update_status(f"Failed loading checkpoint")
            return

        self.load_pdf(pdf_path=ckpt["pdf_path"], current_page=ckpt["page"])
        self.update_status(
            f"Opened PDF file {self.current_pdf_path} from checkpoint")

    def save_ckpt(self) -> None:
        ckpt = {"pdf_path": self.current_pdf_path,
                "page": self.current_page}

        self.file_processor.write_json(ckpt, self.ckpt_path)

    def prev_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self.load_pdf(self.current_pdf_path, self.current_page)
            self.save_ckpt()

    def next_page(self) -> None:
        if self.current_page < self.page_number - 1:
            self.current_page += 1
            self.load_pdf(self.current_pdf_path, self.current_page)
            self.save_ckpt()

    def reset_view(self):
        self.zoom_level = 1.0
        self._resize_and_display_image()

    def update_status(self, message) -> None:
        self.status.config(text=message)
        self.root.update_idletasks()

    def run(self) -> None:
        self.root.mainloop()


class UserInterface:
    def __init__(self, root: tk.Tk, main_app: ExtractorApp) -> None:
        self.root = root
        self.main_app = main_app

        self.submenu_font = ("Segoe UI", 12)

        self.left_width_weight = 3
        self.right_width_weight = 7
        self.right_weights = [1, 1]

    def setup_ui(self) -> None:
        self.setup_root()
        self.setup_menubar()
        self.setup_panes()
        self.setup_status_bar()

    def setup_root(self) -> None:
        # DPI
        ctypes.windll.shcore.SetProcessDpiAwareness(2)

        self.root.title("Testing Digitizer")
        self.root.state('zoomed')

        # Maxsize and minsize
        maxsize = int(self.root.winfo_screenwidth() * 1.25)
        minsize = int(self.root.winfo_screenheight() * 1.25)
        self.root.maxsize(maxsize, minsize)
        self.root.minsize(int(maxsize / 1.5), int(minsize / 1.5))

    def setup_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0, font=self.submenu_font)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.main_app.open_pdf)
        file_menu.add_command(label="Load checkpoints",
                              command=self.main_app.load_ckpt)
        file_menu.add_separator()
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

        # Main container for canvas and scrollbars
        canvas_frame = ttk.Frame(left_frame)

        # Scrollbars
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        self.left_canvas = tk.Canvas(
            canvas_frame,
            bg='lightgray',
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set
        )

        h_scroll.config(command=self.left_canvas.xview)
        v_scroll.config(command=self.left_canvas.yview)

        # Pack canvas and scrollbars in their own frame
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.left_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Pack the canvas frame at top (will expand to fill available space)
        canvas_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Navigation controls at very bottom
        self.setup_navigation_controls(left_frame)

        self.left_canvas.bind('<MouseWheel>', self.main_app.on_mousewheel)

        self.main_pane.add(left_frame, weight=self.left_width_weight)

    def setup_navigation_controls(self, parent) -> None:
        nav_frame = ttk.Frame(parent)
        prev_btn = ttk.Button(nav_frame, text="< Prev",
                              command=self.main_app.prev_page)
        next_btn = ttk.Button(nav_frame, text="Next >",
                              command=self.main_app.next_page)
        reset_btn = ttk.Button(nav_frame, text="Reset view",
                               command=self.main_app.reset_view)

        prev_btn.pack(side=tk.LEFT)
        next_btn.pack(side=tk.LEFT)
        reset_btn.pack(side=tk.LEFT)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

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

        self.main_pane.add(right_frame, weight=self.right_width_weight)

    def setup_status_bar(self) -> None:
        self.main_app.status = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN)
        self.main_app.status.pack(side=tk.BOTTOM, fill=tk.X)
