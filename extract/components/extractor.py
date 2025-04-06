import os
from PIL import Image, ImageTk, ImageDraw
import ctypes
import tkinter as tk
from tkinter import ttk, filedialog
from modules.handlers import PDFHandler, ImageHandler
from modules.processors import FileProcessor
from modules.predictors.segmentation import YOLO_SegmentationPredictor


class ExtractorApp:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.root = tk.Tk()
        self.ui = UserInterface(self.root, self)
        self.ui.setup_ui()

        self.pdf_handler = PDFHandler()
        self.image_handler = ImageHandler()
        self.file_processor = FileProcessor()

        self.inputs_dir = "inputs"
        self.ckpt_path = os.path.join(self.inputs_dir, "checkpoints.json")

        self.current_pdf_path = None
        self.current_pdf_obj = None
        self.current_page = 0
        self.page_count = 0

        self.base_image_dimensions = (595, 842)
        self.zoom_level = 1.0
        self.dragging = False

        self.original_image = None
        self.base_image = None
        self.resized_image = None
        self.tk_image = None
        self.canvas_image = None

        self.colors = self._initialize_colors()
        self.page_predictor = YOLO_SegmentationPredictor(
            cfg["model_path"]["page"])
        self.question_predictor = YOLO_SegmentationPredictor(
            cfg["model_path"]["question"])

    @staticmethod
    def _initialize_colors() -> dict:
        return {
            0: (255, 0, 0, 128),
            1: (0, 255, 0, 128),
            2: (0, 0, 255, 128),
            3: (255, 255, 0, 128),
            4: (255, 0, 255, 128),
            5: (0, 255, 255, 128),
            6: (128, 0, 128, 128),
            7: (255, 165, 0, 128),
        }

    def open_pdf(self) -> None:
        pdf_path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf")])
        if pdf_path:
            self.load_pdf(pdf_path)
            self.save_checkpoint()
            self.update_status(f"Opened PDF file: {pdf_path}")

    def load_pdf(self, pdf_path: str, current_page: int = 0) -> None:
        self.current_pdf_path = pdf_path
        self.current_pdf_obj = self.pdf_handler.open_pdf(pdf_path)
        self.current_page = current_page
        self.page_count = len(self.current_pdf_obj)

        self._load_page_image()

    def _load_page_image(self) -> None:
        pdf_page = self.current_pdf_obj[self.current_page]
        self.original_image = self.pdf_handler.get_page_image(pdf_page)
        self.base_image = self.image_handler.resize_image(
            self.original_image, *self.base_image_dimensions
        )
        self.zoom_level = 1.0
        self._update_canvas_image()

    def _resize_image(self, canvas_width: int, canvas_height: int) -> Image.Image:
        """Resize the image based on the current zoom level or fit it to the canvas."""
        if self.zoom_level == 1.0:
            return self.image_handler.resize_image(self.base_image, canvas_width, canvas_height)
        else:
            width = int(self.base_image.width * self.zoom_level)
            height = int(self.base_image.height * self.zoom_level)
            return self.base_image.resize((width, height), Image.Resampling.LANCZOS)

    def _update_canvas_image(self) -> None:
        """Update the canvas with the resized image."""
        if not self.base_image:
            return

        canvas_width = self.ui.left_canvas.winfo_width()
        canvas_height = self.ui.left_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        self.resized_image = self._resize_image(canvas_width, canvas_height)
        self.tk_image = ImageTk.PhotoImage(self.resized_image)

        if self.canvas_image:
            self.ui.left_canvas.itemconfig(
                self.canvas_image, image=self.tk_image)
        else:
            self.canvas_image = self.ui.left_canvas.create_image(
                0, 0, anchor=tk.NW, image=self.tk_image)

        self.ui.left_canvas.config(
            scrollregion=self.ui.left_canvas.bbox(tk.ALL))

    def zoom_in(self) -> None:
        self.zoom_level *= 1.1
        self._update_canvas_image()

    def zoom_out(self) -> None:
        self.zoom_level /= 1.1
        self._update_canvas_image()

    def navigate_page(self, direction: int) -> None:
        new_page = self.current_page + direction
        if 0 <= new_page < self.page_count:
            self.current_page = new_page
            self._load_page_image()
            self.save_checkpoint()

    def save_checkpoint(self) -> None:
        checkpoint = {"pdf_path": self.current_pdf_path,
                      "page": self.current_page}
        self.file_processor.write_json(checkpoint, self.ckpt_path)

    def load_checkpoint(self) -> None:
        checkpoint = self.file_processor.read_json(self.ckpt_path)
        if checkpoint:
            self.load_pdf(
                pdf_path=checkpoint["pdf_path"], current_page=checkpoint["page"])
            self.update_status(
                f"Opened PDF file {self.current_pdf_path} from checkpoint")
        else:
            self.update_status("Failed loading checkpoint")

    def run_ml(self) -> None:
        if not self.original_image:
            return

        page_predictions = self.page_predictor.predict(self.original_image)
        self._process_predictions(page_predictions)

    def _process_predictions(self, predictions) -> None:
        drawn_image = self._draw_polygons(
            self.original_image, predictions.id2polygons)
        self.base_image = self.image_handler.resize_image(
            drawn_image, *self.base_image_dimensions)

        self.question_images = [
            self.image_handler.crop_image(self.original_image, polygon)
            for cls, polygons in predictions.id2polygons.items()
            if predictions.id2label[cls] == "question"
            for polygon in polygons
        ]

        self.ui.setup_question_controls(len(self.question_images))
        self._update_canvas_image()

    def _draw_polygons(self, image: Image.Image, id2polygons: dict) -> Image.Image:
        drawn_image = image.copy()
        draw = ImageDraw.Draw(drawn_image, "RGBA")
        for cls, polygons in id2polygons.items():
            for polygon in polygons:
                draw.polygon(polygon, fill=self.colors[cls], outline="black")
        return drawn_image

    def update_status(self, message: str) -> None:
        self.ui.update_status(message)

    def run(self) -> None:
        self.root.mainloop()

    def reset_view(self) -> None:
        """Reset the zoom level and update the canvas image."""
        self.zoom_level = 1.0
        self._update_canvas_image()

    def start_drag(self, event: tk.Event) -> None:
        """Start dragging the canvas."""
        self.dragging = True
        self.ui.left_canvas.scan_mark(event.x, event.y)

    def on_drag(self, event: tk.Event) -> None:
        """Handle dragging the canvas."""
        if self.dragging:
            self.ui.left_canvas.scan_dragto(event.x, event.y, gain=1)

    def stop_drag(self, event: tk.Event) -> None:
        """Stop dragging the canvas."""
        self.dragging = False

    def on_mousewheel(self, event: tk.Event) -> None:
        """Handle mouse wheel events for zooming or scrolling."""
        if event.state & 0x0004:  # Check if Control key is pressed
            if event.delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            scroll_amount = -event.delta // 120  # Scroll vertically
            self.ui.left_canvas.yview_scroll(scroll_amount, "units")
        return "break"  # Prevent other handlers from processing the event


class UserInterface:
    def __init__(self, root: tk.Tk, app: ExtractorApp) -> None:
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
        """Create a canvas with horizontal and vertical scrollbars."""
        h_scroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL)
        canvas = tk.Canvas(
            parent, bg="lightgray", xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set
        )
        h_scroll.config(command=canvas.xview)
        v_scroll.config(command=canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Bind events for zooming and dragging
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
    def _resize_image_to_fit_canvas(image: Image.Image, canvas_width: int, canvas_height: int) -> Image.Image:
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
