import sys
import cv2
import numpy as np
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, 
                             QGroupBox, QGridLayout, QSizePolicy, QProgressDialog,
                             QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


# Core Image Processing Logic
class ImageProcessor:
    @staticmethod
    def white_balance(img, a_shift, b_shift):
        """Corrects color cast in LAB color space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        A = A.astype(np.float32)
        B = B.astype(np.float32)

        # Shift A (Green-Red) and B (Blue-Yellow) components
        A = A - (np.mean(A) - 128) + a_shift
        B = B - (np.mean(B) - 128) + b_shift

        A = np.clip(A, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def restore_red(img, strength):
        """Compensates for red light absorption."""
        b, g, r = cv2.split(img)
        # Histogram equalization on the red channel
        boost = cv2.equalizeHist(r)
        strength_factor = strength / 100.0
        # Blend original red with boosted red
        r_new = cv2.addWeighted(r, 1 - strength_factor, boost, strength_factor, 0)
        return cv2.merge([b, g, r_new])

    @staticmethod
    def clahe_enhance(img, clip_limit):
        """Contrast-Limited Adaptive Histogram Equalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        # Limit contrast enhancement to avoid amplifying noise
        clip_val = max(clip_limit / 10.0, 0.1)
        clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def dehaze(img, omega):
        """Dehazing using Dark Channel Prior."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Estimate atmospheric light (top 5% brightest pixels)
        A_light = np.percentile(gray, 95)
        if A_light == 0: A_light = 1
        
        omega_val = omega / 100.0
        t = 1 - omega_val * (gray / A_light)
        t = np.clip(t, 0.35, 1.0) # T_MIN to prevent over-enhancement
        
        t_stack = cv2.merge([t, t, t])
        J = (img.astype(np.float32) - A_light) / t_stack + A_light
        return np.clip(J, 0, 255).astype(np.uint8)

    @staticmethod
    def sharpen(img):
        """Adaptive Unsharp Masking."""
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        return cv2.addWeighted(img, 1.2, blur, -0.2, 0)

    @staticmethod
    def gamma_correct(img, g=1.1):
        """Gamma correction to brighten shadows."""
        inv = 1.0 / g
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def process_pipeline(img, params):
        """Executes the full enhancement pipeline."""
        if img is None: return None
        
        res = ImageProcessor.white_balance(img, params['a_shift'], params['b_shift'])
        res = ImageProcessor.restore_red(res, params['red_strength'])
        res = ImageProcessor.clahe_enhance(res, params['clahe_clip'])
        res = ImageProcessor.dehaze(res, params['omega'])
        res = ImageProcessor.sharpen(res)
        res = ImageProcessor.gamma_correct(res)
        return res


# ---------------------------------------------------------------------------
# Automatic Per-Frame Parameter Tuner
# ---------------------------------------------------------------------------
class AutoTuner:
    """Lightweight coordinate search that optimizes enhancement parameters
    per frame using a no-reference image quality score.

    The scorer rewards contrast and fine detail while penalizing residual
    color cast, excessive saturation, and highlight / shadow clipping.
    The search runs on a heavily downscaled proxy of each frame so that it
    stays fast enough for real-time or near-real-time use.
    """

    # Downscale target width for the proxy image used during search
    PROXY_WIDTH = 160

    # Parameter search space – each tuple is (key, step_size, min, max)
    SEARCH_SPACE = [
        ('a_shift',      5, -50,  50),
        ('b_shift',      5, -50,  50),
        ('red_strength', 10,   0, 100),
        ('clahe_clip',    4,   1,  50),
        ('omega',        10,   0, 100),
    ]

    # Maximum coordinate-descent iterations (passes over all parameters)
    MAX_ITERS = 2

    # ------------------------------------------------------------------
    # Quality score (higher is better)
    # ------------------------------------------------------------------
    @staticmethod
    def score_image(img):
        """Compute a no-reference quality score for a BGR image.

        Components (all operate on float64 for precision):
          +contrast   : standard deviation of luminance (higher = more contrast)
          +detail     : mean of Laplacian magnitude (higher = more detail)
          −color_cast : deviation of mean A and mean B from neutral 128
          −excess_sat : penalty when mean saturation exceeds a comfortable
                        threshold (prevents garish over-saturation)
          −clipping   : fraction of pixels that are fully black or fully white
        """
        # Convert once
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L_f = L.astype(np.float64)

        # Contrast – std dev of lightness
        contrast = np.std(L_f)

        # Detail / sharpness – variance of Laplacian
        lap = cv2.Laplacian(L, cv2.CV_64F)
        detail = np.mean(np.abs(lap))

        # Color-cast penalty – distance of mean chrominance from neutral
        a_mean = np.mean(A.astype(np.float64))
        b_mean = np.mean(B.astype(np.float64))
        color_cast = np.sqrt((a_mean - 128.0) ** 2 + (b_mean - 128.0) ** 2)

        # Excess saturation penalty
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_mean = np.mean(hsv[:, :, 1].astype(np.float64))
        # Comfortable band ~60-130; penalise above 130
        excess_sat = max(0.0, sat_mean - 130.0)

        # Clipping penalty – fraction of pixels at 0 or 255
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n_pixels = gray.size
        clipped = np.sum(gray == 0) + np.sum(gray == 255)
        clip_frac = clipped / n_pixels  # 0..1

        # Weighted combination
        score = (1.0 * contrast
                 + 0.5 * detail
                 - 1.5 * color_cast
                 - 0.8 * excess_sat
                 - 80.0 * clip_frac)
        return score

    # ------------------------------------------------------------------
    # Coordinate search
    # ------------------------------------------------------------------
    @classmethod
    def tune(cls, frame, seed_params):
        """Run a coordinate search starting from *seed_params* on a
        downscaled proxy of *frame*.  Returns the best parameter dict found.

        Parameters
        ----------
        frame : np.ndarray
            Original (or preview-sized) BGR frame.
        seed_params : dict
            Starting parameter values (typically from the GUI sliders).

        Returns
        -------
        dict
            Optimised parameter values.
        """
        # Build proxy
        h, w = frame.shape[:2]
        if w > cls.PROXY_WIDTH:
            scale = cls.PROXY_WIDTH / w
            proxy = cv2.resize(frame, (cls.PROXY_WIDTH, max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            proxy = frame

        best_params = dict(seed_params)  # copy
        best_score = cls._evaluate(proxy, best_params)

        for _ in range(cls.MAX_ITERS):
            improved = False
            for key, step, lo, hi in cls.SEARCH_SPACE:
                current_val = best_params[key]
                for direction in (+step, -step):
                    candidate_val = current_val + direction
                    if candidate_val < lo or candidate_val > hi:
                        continue
                    trial = dict(best_params)
                    trial[key] = candidate_val
                    s = cls._evaluate(proxy, trial)
                    if s > best_score:
                        best_score = s
                        best_params[key] = candidate_val
                        improved = True
            if not improved:
                break  # converged

        return best_params

    @classmethod
    def _evaluate(cls, proxy, params):
        """Process the proxy through the pipeline and score it."""
        result = ImageProcessor.process_pipeline(proxy, params)
        if result is None:
            return -1e9
        return cls.score_image(result)


# PyQt5 Main Window (GUI)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Underwater Image & Video Enhancer")
        self.resize(1280, 850) 

        # Data storage & state
        self.is_video = False
        self.video_path = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_video_frame)

        self.original_image = None  # Full resolution original / frame
        self.preview_image = None   # Downscaled preview frame
        self.processed_image = None # Processed preview frame

        # Auto-tune state
        self.autotune_enabled = False
        self.last_tuned_params = None  # cached result for static images

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        controls_layout = QVBoxLayout()
        controls_group = QGroupBox("Parameters")
        controls_group.setFixedWidth(300)
        controls_inner_layout = QVBoxLayout()

        self.sliders = {}
        self.slider_labels = {}
        # Config: (key, min, max, default, label)
        slider_configs = [
            ('a_shift', -50, 50, 0, "A Shift (Green-Red)"),
            ('b_shift', -50, 50, 0, "B Shift (Blue-Yellow)"),
            ('omega', 0, 100, 75, "Dehaze (Omega)"),
            ('clahe_clip', 1, 50, 12, "Contrast (CLAHE)"),
            ('red_strength', 0, 100, 30, "Red Boost")
        ]

        for key, min_v, max_v, def_v, label_text in slider_configs:
            lbl = QLabel(f"{label_text}: {def_v}")
            sld = QSlider(Qt.Horizontal)
            sld.setMinimum(min_v)
            sld.setMaximum(max_v)
            sld.setValue(def_v)
            sld.valueChanged.connect(lambda val, k=key, l=lbl, txt=label_text: self.on_slider_change(val, k, l, txt))
            
            controls_inner_layout.addWidget(lbl)
            controls_inner_layout.addWidget(sld)
            self.sliders[key] = sld
            self.slider_labels[key] = (lbl, label_text)

        # --- Auto-tune checkbox ---
        self.chk_autotune = QCheckBox("Auto-tune parameters per frame")
        self.chk_autotune.setToolTip(
            "When enabled, a lightweight coordinate search automatically\n"
            "adjusts white-balance shifts, red boost, CLAHE, and dehaze\n"
            "settings for every frame.  Slider values are used as the\n"
            "starting point for the search."
        )
        self.chk_autotune.stateChanged.connect(self.on_autotune_toggled)
        controls_inner_layout.addWidget(self.chk_autotune)

        controls_inner_layout.addStretch()
        controls_group.setLayout(controls_inner_layout)
        
        # Action Buttons
        btn_layout = QVBoxLayout()
        
        self.btn_load = QPushButton("Load Media (Image/Video)")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_media)
        
        self.btn_play = QPushButton("Play / Pause Video")
        self.btn_play.setMinimumHeight(40)
        self.btn_play.clicked.connect(self.toggle_play_video)
        self.btn_play.setEnabled(False)

        self.btn_save = QPushButton("Save Enhanced Result")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(self.save_media)
        self.btn_save.setEnabled(False)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_save)

        controls_layout.addWidget(controls_group)
        controls_layout.addLayout(btn_layout)
        controls_layout.addStretch()

        # Display Area
        display_layout = QGridLayout()
        
        self.lbl_orig_title = QLabel("Original")
        self.lbl_orig_title.setAlignment(Qt.AlignCenter)
        self.lbl_orig_title.setMaximumHeight(30)
        
        self.lbl_proc_title = QLabel("Enhanced")
        self.lbl_proc_title.setAlignment(Qt.AlignCenter)
        self.lbl_proc_title.setMaximumHeight(30)

        self.lbl_orig_img = QLabel()
        self.lbl_orig_img.setAlignment(Qt.AlignCenter)
        self.lbl_orig_img.setStyleSheet("border: 1px solid #cccccc; background-color: #1e1e1e;")
        self.lbl_orig_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.lbl_proc_img = QLabel()
        self.lbl_proc_img.setAlignment(Qt.AlignCenter)
        self.lbl_proc_img.setStyleSheet("border: 1px solid #cccccc; background-color: #1e1e1e;")
        self.lbl_proc_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        display_layout.addWidget(self.lbl_orig_title, 0, 0)
        display_layout.addWidget(self.lbl_proc_title, 0, 1)
        display_layout.addWidget(self.lbl_orig_img, 1, 0)
        display_layout.addWidget(self.lbl_proc_img, 1, 1)
        display_layout.setRowStretch(1, 1)

        main_layout.addLayout(controls_layout)
        main_layout.addLayout(display_layout, stretch=1)

    # ------------------------------------------------------------------
    # Auto-tune helpers
    # ------------------------------------------------------------------
    def on_autotune_toggled(self, state):
        """Called when the auto-tune checkbox is toggled."""
        self.autotune_enabled = (state == Qt.Checked)
        self.last_tuned_params = None  # invalidate cache
        if self.preview_image is not None and not self.timer.isActive():
            self.update_processing()

    def _get_effective_params(self, frame):
        """Return the parameter dict to use for *frame*.

        If auto-tune is disabled, returns slider values directly.
        If auto-tune is enabled, runs the coordinate search using slider
        values as seeds and returns the optimised result.
        """
        seed = self.get_current_params()
        if not self.autotune_enabled:
            return seed
        return AutoTuner.tune(frame, seed)

    def _update_slider_labels_for_tuned(self, params):
        """Update slider labels to reflect the auto-tuned values
        (does NOT move the sliders themselves – they remain as seeds)."""
        for key, val in params.items():
            if key in self.slider_labels:
                lbl, label_text = self.slider_labels[key]
                display_val = val
                if key == 'omega':
                    display_val = val / 100.0
                elif key == 'clahe_clip':
                    display_val = val / 10.0
                lbl.setText(f"{label_text}: {display_val}  [auto]")

    def _restore_slider_labels(self):
        """Restore slider labels to show their manual values (no '[auto]' tag)."""
        for key, sld in self.sliders.items():
            if key in self.slider_labels:
                lbl, label_text = self.slider_labels[key]
                val = sld.value()
                display_val = val
                if key == 'omega':
                    display_val = val / 100.0
                elif key == 'clahe_clip':
                    display_val = val / 10.0
                lbl.setText(f"{label_text}: {display_val}")

    # ------------------------------------------------------------------

    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_video = False
        self.btn_play.setEnabled(False)
        self.btn_play.setText("Play / Pause Video")

    def load_media(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image or Video", "", 
            "All Supported Media (*.jpg *.jpeg *.png *.bmp *.tiff *.mp4 *.avi *.mov *.mkv);;Image Files (*.jpg *.jpeg *.png *.bmp *.tiff);;Video Files (*.mp4 *.avi *.mov *.mkv)", 
            options=options
        )
        if not file_path:
            return

        self.stop_video()
        self.last_tuned_params = None  # invalidate cache for new media
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            self.load_video(file_path)
        else:
            self.load_image(file_path)

    def load_image(self, file_path):
        img_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            return

        self.is_video = False
        self.original_image = img
        self.preview_image = self.downscale_for_preview(img)
        
        self.display_image(self.preview_image, self.lbl_orig_img)
        self.update_processing()
        self.btn_save.setEnabled(True)

    def load_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file.")
            return

        self.is_video = True
        self.video_path = file_path
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        interval = int(1000 / fps) if fps > 0 else 33

        self.btn_play.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_play.setText("Pause Video")

        # Read first frame to initialize displays
        ret, frame = self.cap.read()
        if ret:
            self.original_image = frame
            self.preview_image = self.downscale_for_preview(frame)
            self.display_image(self.preview_image, self.lbl_orig_img)
            self.update_processing()

        # Start playing
        self.timer.start(interval)

    def downscale_for_preview(self, img):
        if img is None: return None
        h, w = img.shape[:2]
        max_display_w = 960
        if w > max_display_w:
            scale = max_display_w / w
            return cv2.resize(img, (int(w * scale), int(h * scale)))
        return img.copy()

    def toggle_play_video(self):
        if not self.is_video or self.cap is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play Video")
        else:
            self.timer.start()
            self.btn_play.setText("Pause Video")

    def next_video_frame(self):
        if not self.is_video or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Loop video playback
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return

        self.original_image = frame
        self.preview_image = self.downscale_for_preview(frame)
        self.display_image(self.preview_image, self.lbl_orig_img)
        self.update_processing()

    def on_slider_change(self, value, key, label_obj, label_text):
        display_val = value
        if key == 'omega': display_val = value / 100.0
        if key == 'clahe_clip': display_val = value / 10.0
        
        label_obj.setText(f"{label_text}: {display_val}")
        # Invalidate tuned-param cache when user adjusts a slider
        self.last_tuned_params = None
        if self.preview_image is not None and not self.timer.isActive():
            self.update_processing()

    def get_current_params(self):
        return {k: v.value() for k, v in self.sliders.items()}

    def update_processing(self):
        if self.preview_image is None:
            return
        
        params = self._get_effective_params(self.preview_image)
        # Cache tuned params for save operations on static images
        if self.autotune_enabled:
            self.last_tuned_params = params
            self._update_slider_labels_for_tuned(params)
        else:
            self._restore_slider_labels()

        processed = ImageProcessor.process_pipeline(self.preview_image, params)
        self.processed_image = processed
        self.display_image(processed, self.lbl_proc_img)

    def display_image(self, img, label_widget):
        if img is None: return

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        label_size = label_widget.size()
        if label_size.width() < 10 or label_size.height() < 10:
            label_size = self.size() 

        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)

    def save_media(self):
        if self.is_video:
            self.save_video()
        else:
            self.save_image()

    def save_image(self):
        if self.original_image is None:
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "enhanced_result.jpg", 
                                                   "Image Files (*.jpg *.png)", options=options)
        if file_path:
            print("Processing full resolution image, please wait...")
            # For images, re-tune on the full-res frame (or use cached result)
            if self.autotune_enabled:
                params = AutoTuner.tune(self.original_image, self.get_current_params())
            else:
                params = self.get_current_params()
            final_result = ImageProcessor.process_pipeline(self.original_image, params)
            
            ext = os.path.splitext(file_path)[1]
            if not ext: ext = ".jpg"
            success, encoded_img = cv2.imencode(ext, final_result)
            if success:
                encoded_img.tofile(file_path)
                print(f"Saved successfully: {file_path}")
            else:
                print("Failed to save image.")

    def save_video(self):
        if not self.video_path:
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Enhanced Video", "enhanced_video.mp4", 
                                                   "Video Files (*.mp4 *.avi)", options=options)
        if not file_path:
            return

        was_playing = self.timer.isActive()
        if was_playing:
            self.timer.stop()

        vcap = cv2.VideoCapture(self.video_path)
        if not vcap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video source for export.")
            return

        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if file_path.endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        seed_params = self.get_current_params()
        label = ("Processing video frames (auto-tuning each)..."
                 if self.autotune_enabled
                 else "Processing video frames...")
        progress = QProgressDialog(label, "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        frame_idx = 0
        while True:
            ret, frame = vcap.read()
            if not ret or progress.wasCanceled():
                break

            if self.autotune_enabled:
                params = AutoTuner.tune(frame, seed_params)
            else:
                params = seed_params

            enhanced_frame = ImageProcessor.process_pipeline(frame, params)
            out.write(enhanced_frame)

            frame_idx += 1
            progress.setValue(frame_idx)
            QApplication.processEvents()

        vcap.release()
        out.release()
        progress.setValue(total_frames)

        if not progress.wasCanceled():
            QMessageBox.information(self, "Success", f"Video enhanced and saved successfully to:\n{file_path}")
            print(f"Video saved successfully: {file_path}")

        if was_playing:
            self.timer.start()

    def resizeEvent(self, event):
        if self.preview_image is not None:
            self.display_image(self.preview_image, self.lbl_orig_img)
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.lbl_proc_img)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())