import os
from PIL import ImageTk
import tkinter as tk
from tkinter import filedialog
from modules.handlers import PDFHandler, ImageHandler
from modules.processors import FileProcessor
from components.ui import UserInterface
from components.managers import PDFManager, ImageManager, PredictionManager  # Updated import


class ExtractorApp:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.root = tk.Tk()
        self.ui = UserInterface(self.root, self)
        self.ui.setup_ui()

        self.pdf_manager = PDFManager(PDFHandler(), FileProcessor(), "inputs")
        self.image_manager = ImageManager(ImageHandler(), (1525, 2048))
        self.prediction_manager = PredictionManager(cfg, ImageHandler())

        self.zoom_level = 1.0
        self.dragging = False
        self.question_images = []  # Add this attribute to store question images

    def open_pdf(self) -> None:
        pdf_path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf")])
        if pdf_path:
            self.pdf_manager.open_pdf(pdf_path)
            self._load_page_image()
            self.pdf_manager.save_checkpoint()
            self.question_images = []
            self.prediction_manager.processed_question_images = []
            self.ui.setup_question_controls(0)
            self.ui.clear_top_canvas()
            self.update_status(f"Opened PDF file: {pdf_path}")

    def _load_page_image(self) -> None:
        pdf_page = self.pdf_manager.current_pdf_obj[self.pdf_manager.current_page]
        self.image_manager.load_page_image(pdf_page)
        self.zoom_level = 1.0
        self._update_canvas_image()

    def _update_canvas_image(self) -> None:
        canvas_width = self.ui.left_canvas.winfo_width()
        canvas_height = self.ui.left_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        self.image_manager.resized_image = self.image_manager.resize_image(
            self.zoom_level, canvas_width, canvas_height
        )
        self.image_manager.tk_image = ImageTk.PhotoImage(
            self.image_manager.resized_image)

        if hasattr(self, "canvas_image") and self.canvas_image:
            self.ui.left_canvas.itemconfig(
                self.canvas_image, image=self.image_manager.tk_image)
        else:
            self.canvas_image = self.ui.left_canvas.create_image(
                0, 0, anchor=tk.NW, image=self.image_manager.tk_image
            )

        self.ui.left_canvas.config(
            scrollregion=self.ui.left_canvas.bbox(tk.ALL))

    def zoom_in(self) -> None:
        self.zoom_level *= 1.1
        self._update_canvas_image()

    def zoom_out(self) -> None:
        self.zoom_level /= 1.1
        self._update_canvas_image()

    def navigate_page(self, direction: int) -> None:
        new_page = self.pdf_manager.current_page + direction
        if 0 <= new_page < self.pdf_manager.page_count:
            self.pdf_manager.current_page = new_page
            self._load_page_image()
            self.pdf_manager.save_checkpoint()
            self.question_images = []
            self.prediction_manager.processed_question_images = []
            self.ui.setup_question_controls(0)
            self.ui.clear_top_canvas()

            # Update status after navigating pages
            self.update_status(
                f"Navigated to page {new_page + 1} of {self.pdf_manager.page_count}.")

    def run_ml(self) -> None:
        if not self.image_manager.original_image:
            return

        drawn_image, num_questions = self.prediction_manager.run_ml(
            self.image_manager.original_image
        )
        self.image_manager.base_image = self.image_manager.image_handler.resize_image(
            drawn_image, *self.image_manager.base_image_dimensions
        )
        self.question_images = self.prediction_manager.question_images
        self.ui.setup_question_controls(num_questions)
        self._update_canvas_image()

        if self.prediction_manager.processed_question_images:  # Display the first processed question image
            self.ui.display_question_image(0)

        # Update status after running ML
        self.update_status(
            f"ML processing completed. {num_questions} questions detected.")

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

    def load_checkpoint(self) -> None:
        checkpoint = self.pdf_manager.load_checkpoint()
        if checkpoint:
            self.pdf_manager.open_pdf(checkpoint["pdf_path"])
            self.pdf_manager.current_page = checkpoint["page"]
            self._load_page_image()
            self.update_status(
                f"Opened PDF file {self.pdf_manager.current_pdf_path} from checkpoint"
            )
        else:
            self.update_status("Failed loading checkpoint")

    def save_page_image(self) -> None:
        if not self.image_manager.original_image:
            self.update_status("No page image to save.")
            return

        pdf_filename = os.path.splitext(
            os.path.basename(self.pdf_manager.current_pdf_path))[0]
        save_dir = self.cfg["train_data_path"]["page"]
        os.makedirs(save_dir, exist_ok=True)
        page_filename = f"{pdf_filename}_{self.pdf_manager.current_page}.jpg"
        save_path = os.path.join(save_dir, page_filename)
        self.image_manager.original_image.save(save_path)
        self.update_status(f"Page image saved to {save_path}.")

    def save_question_image(self) -> None:
        """Save the currently selected question image to the configured directory."""
        if not self.prediction_manager.processed_question_images:
            self.update_status("No question image to save.")
            return

        selected_index = self.ui.selected_question_index
        if selected_index < 0 or selected_index >= len(self.prediction_manager.processed_question_images):
            self.update_status("Invalid question index selected.")
            return

        pdf_filename = os.path.splitext(
            os.path.basename(self.pdf_manager.current_pdf_path))[0]
        save_dir = self.cfg["train_data_path"]["question"]
        os.makedirs(save_dir, exist_ok=True)
        question_filename = f"{pdf_filename}_{self.pdf_manager.current_page}_{selected_index}.jpg"
        save_path = os.path.join(save_dir, question_filename)
        self.prediction_manager.question_images[selected_index].save(save_path)

        self.update_status(f"Question image saved to {save_path}.")
