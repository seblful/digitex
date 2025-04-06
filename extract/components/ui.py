import ctypes
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class UserInterface:
    def __init__(self, root: tk.Tk, app) -> None:
        self.root = root
        self.app = app
        self.submenu_font = ("Segoe UI", 12)
        self.left_width_weight = 3
        self.right_width_weight = 7

    def setup_ui(self) -> None:
        self._setup_root()
        self._setup_menubar()
        self._setup_panes()
        self._setup_status_bar()

    def _setup_root(self) -> None:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        self.root.title("Testing Digitizer")
        self.root.state("zoomed")
        screen_width, screen_height = self.root.winfo_screenwidth(
        ), self.root.winfo_screenheight()
        self.root.maxsize(int(screen_width * 1.25), int(screen_height * 1.25))
        self.root.minsize(int(screen_width * 0.8), int(screen_height * 0.8))

    def _setup_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0, font=self.submenu_font)
        file_menu.add_command(label="Open", command=self.app.open_pdf)
        file_menu.add_command(label="Load Checkpoint",
                              command=self.app.load_checkpoint)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _setup_panes(self) -> None:
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(expand=True, fill=tk.BOTH)
        self._setup_left_pane()
        self._setup_right_pane()

    def _setup_left_pane(self) -> None:
        left_frame = ttk.Frame(self.main_pane)
        canvas_frame = ttk.Frame(left_frame)
        self.left_canvas = self._create_canvas_with_scrollbars(canvas_frame)
        canvas_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        self._setup_navigation_controls(left_frame)
        self.main_pane.add(left_frame, weight=self.left_width_weight)

    def _create_canvas_with_scrollbars(self, parent: ttk.Frame) -> tk.Canvas:
        h_scroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL)
        canvas = tk.Canvas(
            parent, bg="lightgray", xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        h_scroll.config(command=canvas.xview)
        v_scroll.config(command=canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        canvas.bind("<MouseWheel>", self.app.on_mousewheel)
        canvas.bind("<ButtonPress-1>", self.app.start_drag)
        canvas.bind("<B1-Motion>", self.app.on_drag)
        canvas.bind("<ButtonRelease-1>", self.app.stop_drag)

        return canvas

    def _setup_navigation_controls(self, parent: ttk.Frame) -> None:
        nav_frame = ttk.Frame(parent)
        ttk.Button(nav_frame, text="< Prev",
                   command=lambda: self.app.navigate_page(-1)).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next >",
                   command=lambda: self.app.navigate_page(1)).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Reset View",
                   command=self.app.reset_view).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Run ML",
                   command=self.app.run_ml).pack(side=tk.LEFT)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_right_pane(self) -> None:
        right_frame = ttk.Frame(self.main_pane)
        right_pane = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_pane.pack(expand=True, fill=tk.BOTH)

        self._setup_top_frame(right_pane)
        self._setup_bottom_frame(right_pane)

        self.main_pane.add(right_frame, weight=self.right_width_weight)

    def _setup_top_frame(self, parent: ttk.PanedWindow) -> None:
        top_frame = ttk.Frame(parent)
        self.question_nav_frame = ttk.Frame(top_frame)
        self.question_nav_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.top_canvas = tk.Canvas(top_frame, bg="lightgray")
        self.top_canvas.pack(expand=True, fill=tk.BOTH)
        parent.add(top_frame, weight=1)

    def _setup_bottom_frame(self, parent: ttk.PanedWindow) -> None:
        bottom_frame = ttk.Frame(parent)
        bottom_canvas = tk.Canvas(bottom_frame, bg="lightgray")
        bottom_canvas.pack(expand=True, fill=tk.BOTH)
        parent.add(bottom_frame, weight=2)

    def setup_question_controls(self, num_questions: int) -> None:
        for widget in self.question_nav_frame.winfo_children():
            widget.destroy()

        for i in range(1, num_questions + 1):
            ttk.Button(self.question_nav_frame, text=str(
                i), command=lambda idx=i - 1: self.display_question_image(idx)).pack(side=tk.LEFT)

    def display_question_image(self, index: int) -> None:
        self.top_canvas.delete("all")
        question_image = self.app.question_images[index]
        canvas_width, canvas_height = self.top_canvas.winfo_width(
        ), self.top_canvas.winfo_height()
        resized_image = self._resize_image_to_fit_canvas(
            question_image, canvas_width, canvas_height)
        tk_image = ImageTk.PhotoImage(resized_image)
        x_offset = (canvas_width - resized_image.width) // 2
        y_offset = (canvas_height - resized_image.height) // 2
        self.top_canvas.create_image(
            x_offset, y_offset, anchor=tk.NW, image=tk_image)
        self.top_canvas.image = tk_image

    @staticmethod
    def _resize_image_to_fit_canvas(image, canvas_width, canvas_height):
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_dimensions = (int(img_width * scale), int(img_height * scale))
        return image.resize(new_dimensions, Image.Resampling.LANCZOS)

    def _setup_status_bar(self) -> None:
        self.status_label = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message: str) -> None:
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def clear_top_canvas(self) -> None:
        """Clear the right top frame canvas."""
        self.top_canvas.delete("all")
