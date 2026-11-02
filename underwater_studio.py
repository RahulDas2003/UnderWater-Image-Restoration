"""
AquaVision Pro - Professional Underwater Image & Video Enhancement Studio
Marketable, commercial-grade PyQt5 application for digital image processing.
"""

import sys
import cv2
import numpy as np
import os
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSlider, QPushButton, QFileDialog, QGroupBox, QGridLayout, 
    QSizePolicy, QProgressDialog, QMessageBox, QTabWidget, QComboBox, 
    QSpinBox, QDoubleSpinBox, QSplitter, QFrame, QStatusBar, QToolBar, QAction,
    QListWidget, QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect, QThread
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont, QIcon, QLinearGradient, QBrush


# ==============================================================================
# CORE IMAGE & VIDEO PROCESSING PIPELINE
# ==============================================================================
class ImageProcessor:
    @staticmethod
    def white_balance(img, a_shift, b_shift):
        """Color balance correction in LAB color space."""
        if img is None: return None
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        A = A.astype(np.float32)
        B = B.astype(np.float32)

        A = A - (np.mean(A) - 128) + a_shift
        B = B - (np.mean(B) - 128) + b_shift

        A = np.clip(A, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def restore_red(img, strength):
        """Compensates for underwater red light attenuation."""
        if img is None or strength <= 0: return img
        b, g, r = cv2.split(img)
        boost = cv2.equalizeHist(r)
        strength_factor = strength / 100.0
        r_new = cv2.addWeighted(r, 1.0 - strength_factor, boost, strength_factor, 0)
        return cv2.merge([b, g, r_new])

    @staticmethod
    def clahe_enhance(img, clip_limit):
        """Contrast-Limited Adaptive Histogram Equalization."""
        if img is None: return None
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clip_val = max(clip_limit / 10.0, 0.1)
        clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def dehaze(img, omega):
        """Dark Channel Prior Dehazing."""
        if img is None or omega <= 0: return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        A_light = np.percentile(gray, 95)
        if A_light == 0: A_light = 1.0

        omega_val = omega / 100.0
        t = 1.0 - omega_val * (gray / A_light)
        t = np.clip(t, 0.35, 1.0)

        t_stack = cv2.merge([t, t, t])
        J = (img.astype(np.float32) - A_light) / t_stack + A_light
        return np.clip(J, 0, 255).astype(np.uint8)

    @staticmethod
    def denoise(img, strength):
        """Edge-preserving Bilateral Filter Denoising to fix speckle noise."""
        if img is None or strength <= 0: return img
        d = int(max(3, strength // 10))
        sigma_color = strength * 1.5
        sigma_space = strength * 1.5
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    @staticmethod
    def color_adjust(img, saturation, temperature):
        """Fine-tunes color warmth & saturation intensity."""
        if img is None: return None
        res = img.copy()
        if saturation != 100:
            hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= (saturation / 100.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if temperature != 0:
            # Shift blue/red ratio for warmth
            b, g, r = cv2.split(res.astype(np.float32))
            r += temperature * 0.8
            b -= temperature * 0.8
            r = np.clip(r, 0, 255)
            b = np.clip(b, 0, 255)
            res = cv2.merge([b, g, r]).astype(np.uint8)

        return res

    @staticmethod
    def sharpen(img, amount):
        """Adaptive unsharp masking for edge detail enhancement."""
        if img is None or amount <= 0: return img
        factor = amount / 50.0
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        return cv2.addWeighted(img, 1.0 + factor, blur, -factor, 0)

    @staticmethod
    def gamma_correct(img, gamma_val):
        """Gamma correction for shadow detail recovery."""
        if img is None or gamma_val == 1.0: return img
        inv = 1.0 / max(gamma_val, 0.1)
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def process_pipeline(img, params):
        """Executes full modular enhancement pipeline."""
        if img is None: return None
        res = ImageProcessor.white_balance(img, params.get('a_shift', 0), params.get('b_shift', 0))
        res = ImageProcessor.restore_red(res, params.get('red_strength', 30))
        res = ImageProcessor.clahe_enhance(res, params.get('clahe_clip', 12))
        res = ImageProcessor.dehaze(res, params.get('omega', 75))
        
        if params.get('denoise_strength', 0) > 0:
            res = ImageProcessor.denoise(res, params['denoise_strength'])
            
        res = ImageProcessor.color_adjust(res, params.get('saturation', 100), params.get('temperature', 0))
        res = ImageProcessor.sharpen(res, params.get('sharpen_amount', 10))
        res = ImageProcessor.gamma_correct(res, params.get('gamma', 1.1))
        return res

    @staticmethod
    def generate_histogram(img, width=280, height=100):
        """Generates a styled RGB color distribution histogram image."""
        if img is None: return None
        hist_canvas = np.zeros((height, width, 3), dtype=np.uint8) + 20 # Dark BG
        colors = [(255, 80, 80), (80, 255, 80), (80, 150, 255)] # B, G, R
        
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, height - 10, cv2.NORM_MINMAX)
            hist_flat = hist.ravel()
            pts = []
            bin_w = width / 256.0
            for x in range(256):
                val = float(hist_flat[x])
                y = height - 5 - int(val)
                pts.append((int(x * bin_w), y))
            
            for j in range(len(pts) - 1):
                cv2.line(hist_canvas, pts[j], pts[j+1], col, 1, cv2.LINE_AA)

        return hist_canvas


# ==============================================================================
# INTERACTIVE SPLIT-SCREEN & BEFORE/AFTER COMPARISON WIDGET
# ==============================================================================
class SplitCompareWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.orig_pixmap = None
        self.proc_pixmap = None
        self.split_position = 0.5  # 0.0 to 1.0 ratio
        self.is_dragging = False
        self.mode = "side_by_side"  # "side_by_side" or "split_slider"
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #0F172A; border-radius: 8px;")

    def set_images(self, orig_bgr, proc_bgr):
        if orig_bgr is not None:
            self.orig_pixmap = self.bgr_to_pixmap(orig_bgr)
        if proc_bgr is not None:
            self.proc_pixmap = self.bgr_to_pixmap(proc_bgr)
        self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.update()

    def bgr_to_pixmap(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(15, 23, 42)) # Dark slate bg

        if self.orig_pixmap is None or self.proc_pixmap is None:
            painter.setPen(QColor(148, 163, 184))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "📷 Load an image or video to begin enhancement")
            return

        if self.mode == "side_by_side":
            half_w = w // 2 - 4
            # Draw Original Left
            scaled_orig = self.orig_pixmap.scaled(half_w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            ox = (half_w - scaled_orig.width()) // 2
            oy = (h - scaled_orig.height()) // 2
            painter.drawPixmap(ox, oy, scaled_orig)

            # Draw Processed Right
            scaled_proc = self.proc_pixmap.scaled(half_w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            px = half_w + 8 + (half_w - scaled_proc.width()) // 2
            py = (h - scaled_proc.height()) // 2
            painter.drawPixmap(px, py, scaled_proc)

            # Labels
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.fillRect(ox + 10, oy + 10, 80, 24, QColor(0, 0, 0, 160))
            painter.drawText(ox + 10, oy + 10, 80, 24, Qt.AlignCenter, "ORIGINAL")

            painter.fillRect(px + 10, py + 10, 90, 24, QColor(2, 132, 199, 200))
            painter.drawText(px + 10, py + 10, 90, 24, Qt.AlignCenter, "ENHANCED")

            # Separator Line
            painter.setPen(QPen(QColor(51, 65, 85), 2))
            painter.drawLine(half_w + 4, 0, half_w + 4, h)

        elif self.mode == "split_slider":
            # Draw Processed as base frame
            scaled_proc = self.proc_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_orig = self.orig_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            img_w, img_h = scaled_proc.width(), scaled_proc.height()
            offset_x = (w - img_w) // 2
            offset_y = (h - img_h) // 2

            # Draw Processed (Right side portion)
            painter.drawPixmap(offset_x, offset_y, scaled_proc)

            # Draw Original (Left side portion clipped to split position)
            split_x = offset_x + int(img_w * self.split_position)
            clip_rect = QRect(offset_x, offset_y, split_x - offset_x, img_h)
            painter.setClipRect(clip_rect)
            painter.drawPixmap(offset_x, offset_y, scaled_orig)
            painter.setClipping(False)

            # Draw Divider & Drag Handle
            painter.setPen(QPen(QColor(6, 182, 212), 3))
            painter.drawLine(split_x, offset_y, split_x, offset_y + img_h)

            # Handle knob
            painter.setBrush(QBrush(QColor(6, 182, 212)))
            painter.drawEllipse(QPoint(split_x, offset_y + img_h // 2), 14, 14)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(split_x - 10, offset_y + img_h // 2 - 10, 20, 20, Qt.AlignCenter, "↔")

            # Overlay Labels
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.fillRect(offset_x + 10, offset_y + 10, 80, 24, QColor(0, 0, 0, 160))
            painter.drawText(offset_x + 10, offset_y + 10, 80, 24, Qt.AlignCenter, "BEFORE")

            painter.fillRect(offset_x + img_w - 100, offset_y + 10, 90, 24, QColor(2, 132, 199, 200))
            painter.drawText(offset_x + img_w - 100, offset_y + 10, 90, 24, Qt.AlignCenter, "AFTER")

    def mousePressEvent(self, event):
        if self.mode == "split_slider" and event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.update_split_pos(event.x())

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.update_split_pos(event.x())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    def update_split_pos(self, mouse_x):
        w = self.width()
        if w > 0:
            self.split_position = max(0.02, min(0.98, mouse_x / float(w)))
            self.update()


# ==============================================================================
# MAIN STUDIO APPLICATION WINDOW
# ==============================================================================
class UnderwaterStudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AquaVision Studio Pro - Underwater Image & Video Enhancement")
        self.resize(1400, 900)

        # Media & Video State
        self.is_video = False
        self.video_path = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_video_frame)

        self.original_frame = None
        self.preview_frame = None
        self.processed_frame = None

        self.presets = {
            "Custom": {},
            "🌊 Deep Blue Ocean": {'a_shift': 15, 'b_shift': -10, 'red_strength': 55, 'clahe_clip': 18, 'omega': 80, 'denoise_strength': 10, 'sharpen_amount': 15, 'gamma': 1.15, 'saturation': 110, 'temperature': 8},
            "🌿 Coastal Green": {'a_shift': -15, 'b_shift': 20, 'red_strength': 40, 'clahe_clip': 15, 'omega': 70, 'denoise_strength': 8, 'sharpen_amount': 12, 'gamma': 1.1, 'saturation': 105, 'temperature': 5},
            "🤿 Shallow Reef": {'a_shift': 5, 'b_shift': 5, 'red_strength': 30, 'clahe_clip': 10, 'omega': 50, 'denoise_strength': 5, 'sharpen_amount': 10, 'gamma': 1.05, 'saturation': 120, 'temperature': 3},
            "🔍 High Clarity Inspection": {'a_shift': 10, 'b_shift': 0, 'red_strength': 45, 'clahe_clip': 25, 'omega': 85, 'denoise_strength': 15, 'sharpen_amount': 25, 'gamma': 1.2, 'saturation': 100, 'temperature': 0}
        }

        self.apply_dark_theme()
        self.init_ui()

    def apply_dark_theme(self):
        """Applies a high-end dark slate & cyan commercial UI stylesheet."""
        qss = """
        QMainWindow { background-color: #0B0F19; }
        QWidget { color: #E2E8F0; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
        
        QGroupBox {
            background-color: #151D2A;
            border: 1px solid #1E293B;
            border-radius: 8px;
            margin-top: 12px;
            font-weight: bold;
            font-size: 13px;
            color: #38BDF8;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 8px;
            background-color: #151D2A;
            border-radius: 4px;
        }

        QSlider::groove:horizontal {
            border: none;
            height: 6px;
            background: #1E293B;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 #06B6D4);
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #F8FAFC;
            border: 2px solid #06B6D4;
            width: 16px;
            margin-top: -5px;
            margin-bottom: -5px;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #06B6D4;
        }

        QPushButton {
            background-color: #1E293B;
            border: 1px solid #334155;
            color: #F8FAFC;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #0284C7;
            border-color: #38BDF8;
        }
        QPushButton:pressed {
            background-color: #0369A1;
        }
        QPushButton:disabled {
            background-color: #0F172A;
            color: #475569;
            border-color: #1E293B;
        }

        QPushButton#primaryBtn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 #06B6D4);
            border: none;
            color: #FFFFFF;
            font-weight: bold;
        }
        QPushButton#primaryBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369A1, stop:1 #0891B2);
        }

        QComboBox {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 6px 12px;
            color: #F8FAFC;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #151D2A;
            selection-background-color: #0284C7;
            color: #F8FAFC;
            border: 1px solid #334155;
        }

        QStatusBar {
            background-color: #0F172A;
            color: #94A3B8;
            border-top: 1px solid #1E293B;
        }

        QTabWidget::pane { border: 1px solid #1E293B; border-radius: 6px; background-color: #151D2A; }
        QTabBar::tab {
            background: #0F172A;
            color: #94A3B8;
            padding: 8px 16px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #151D2A;
            color: #38BDF8;
            font-weight: bold;
            border-bottom: 2px solid #06B6D4;
        }
        """
        self.setStyleSheet(qss)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QHBoxLayout(main_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # ----------------------------------------------------------------------
        # LEFT CONTROL PANEL
        # ----------------------------------------------------------------------
        panel_layout = QVBoxLayout()
        panel_widget = QWidget()
        panel_widget.setFixedWidth(340)
        panel_widget.setLayout(panel_layout)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        # Header Title
        app_header = QLabel("⚡ AquaVision Pro Studio")
        app_header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        app_header.setStyleSheet("color: #06B6D4; margin-bottom: 4px;")
        panel_layout.addWidget(app_header)

        # Presets Group
        preset_group = QGroupBox("Enhancement Profiles")
        preset_layout = QVBoxLayout()
        self.cb_presets = QComboBox()
        self.cb_presets.addItems(list(self.presets.keys()))
        self.cb_presets.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.cb_presets)
        preset_group.setLayout(preset_layout)
        panel_layout.addWidget(preset_group)

        # Tabbed Parameter Controls
        tabs = QTabWidget()

        # Tab 1: Color & Light
        tab_color = QWidget()
        tab_color_layout = QVBoxLayout(tab_color)
        self.sliders = {}

        self.add_slider(tab_color_layout, 'red_strength', 0, 100, 35, "🔴 Red Boost Strength")
        self.add_slider(tab_color_layout, 'a_shift', -50, 50, 0, "🟢🔴 A Shift (Green-Red)")
        self.add_slider(tab_color_layout, 'b_shift', -50, 50, 0, "🔵🟡 B Shift (Blue-Yellow)")
        self.add_slider(tab_color_layout, 'saturation', 50, 200, 100, "🎨 Color Saturation (%)")
        self.add_slider(tab_color_layout, 'temperature', -30, 30, 0, "🌡️ Warmth / Temp")
        tab_color_layout.addStretch()
        tabs.addTab(tab_color, "Color & Red")

        # Tab 2: Clarity & Contrast
        tab_clarity = QWidget()
        tab_clarity_layout = QVBoxLayout(tab_clarity)
        self.add_slider(tab_clarity_layout, 'omega', 0, 100, 75, "🌫️ Dehaze Strength")
        self.add_slider(tab_clarity_layout, 'clahe_clip', 1, 50, 14, "🌓 Adaptive CLAHE")
        self.add_slider(tab_clarity_layout, 'denoise_strength', 0, 30, 8, "✨ Denoise Speckles")
        self.add_slider(tab_clarity_layout, 'sharpen_amount', 0, 40, 10, "🗡️ Edge Sharpening")
        self.add_slider(tab_clarity_layout, 'gamma', 50, 200, 110, "☀️ Shadow Gamma")
        tab_clarity_layout.addStretch()
        tabs.addTab(tab_clarity, "Clarity & Detail")

        panel_layout.addWidget(tabs)

        # Histogram Live View
        hist_group = QGroupBox("Live RGB Histogram")
        hist_layout = QVBoxLayout()
        self.lbl_hist = QLabel()
        self.lbl_hist.setFixedHeight(90)
        self.lbl_hist.setStyleSheet("background-color: #101522; border-radius: 4px;")
        self.lbl_hist.setAlignment(Qt.AlignCenter)
        hist_layout.addWidget(self.lbl_hist)
        hist_group.setLayout(hist_layout)
        panel_layout.addWidget(hist_group)

        # Buttons Panel
        btn_box = QVBoxLayout()

        self.btn_load = QPushButton("📁 Open Image / Video")
        self.btn_load.setObjectName("primaryBtn")
        self.btn_load.setMinimumHeight(38)
        self.btn_load.clicked.connect(self.load_media)
        btn_box.addWidget(self.btn_load)

        video_controls = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play / Pause")
        self.btn_play.clicked.connect(self.toggle_play_video)
        self.btn_play.setEnabled(False)

        self.btn_snapshot = QPushButton("📸 Snapshot")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.btn_snapshot.setEnabled(False)
        video_controls.addWidget(self.btn_play)
        video_controls.addWidget(self.btn_snapshot)
        btn_box.addLayout(video_controls)

        self.btn_save = QPushButton("💾 Export Enhanced Result")
        self.btn_save.setMinimumHeight(36)
        self.btn_save.clicked.connect(self.export_media)
        self.btn_save.setEnabled(False)
        btn_box.addWidget(self.btn_save)

        self.btn_batch = QPushButton("📦 Batch Process Folder...")
        self.btn_batch.clicked.connect(self.batch_process_folder)
        btn_box.addWidget(self.btn_batch)

        panel_layout.addLayout(btn_box)

        # Add left panel to root
        root_layout.addWidget(panel_widget)

        # ----------------------------------------------------------------------
        # RIGHT DISPLAY AREA & VIEW MODES
        # ----------------------------------------------------------------------
        right_layout = QVBoxLayout()
        
        # Toolbar / Mode Switcher
        toolbar = QHBoxLayout()
        mode_label = QLabel("View Mode:")
        mode_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        
        self.cb_view_mode = QComboBox()
        self.cb_view_mode.addItems(["Side-by-Side Comparison", "Split Slider (Before/After)"])
        self.cb_view_mode.currentIndexChanged.connect(self.on_view_mode_changed)

        self.lbl_info = QLabel("Ready")
        self.lbl_info.setStyleSheet("color: #64748B; font-weight: 500;")

        toolbar.addWidget(mode_label)
        toolbar.addWidget(self.cb_view_mode)
        toolbar.addStretch()
        toolbar.addWidget(self.lbl_info)
        right_layout.addLayout(toolbar)

        # Canvas Display Widget
        self.display_widget = SplitCompareWidget()
        right_layout.addWidget(self.display_widget, stretch=1)

        root_layout.addLayout(right_layout, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("AquaVision Studio Pro initialized.")

    def add_slider(self, layout, key, min_v, max_v, def_v, label_text):
        container = QWidget()
        v_box = QVBoxLayout(container)
        v_box.setContentsMargins(0, 4, 0, 4)

        header = QHBoxLayout()
        lbl_title = QLabel(label_text)
        lbl_val = QLabel(str(def_v))
        lbl_val.setStyleSheet("color: #06B6D4; font-weight: bold;")
        header.addWidget(lbl_title)
        header.addStretch()
        header.addWidget(lbl_val)

        sld = QSlider(Qt.Horizontal)
        sld.setMinimum(min_v)
        sld.setMaximum(max_v)
        sld.setValue(def_v)
        sld.valueChanged.connect(lambda v, k=key, l=lbl_val: self.on_slider_moved(v, k, l))

        v_box.addLayout(header)
        v_box.addWidget(sld)
        layout.addWidget(container)
        self.sliders[key] = (sld, lbl_val)

    def on_slider_moved(self, val, key, label_obj):
        display_val = val
        if key in ['gamma', 'saturation']: display_val = f"{val / 100.0:.2f}"
        if key == 'omega': display_val = f"{val / 100.0:.2f}"
        if key == 'clahe_clip': display_val = f"{val / 10.0:.1f}"
        label_obj.setText(str(display_val))

        if self.cb_presets.currentText() != "Custom":
            self.cb_presets.blockSignals(True)
            self.cb_presets.setCurrentText("Custom")
            self.cb_presets.blockSignals(False)

        if self.preview_frame is not None and not self.timer.isActive():
            self.update_enhancement()

    def on_preset_changed(self, preset_name):
        if preset_name not in self.presets or not self.presets[preset_name]:
            return
        
        params = self.presets[preset_name]
        for k, v in params.items():
            if k in self.sliders:
                sld, lbl = self.sliders[k]
                sld.blockSignals(True)
                sld.setValue(v)
                display_val = v
                if k in ['gamma', 'saturation']: display_val = f"{v / 100.0:.2f}"
                if k == 'omega': display_val = f"{v / 100.0:.2f}"
                if k == 'clahe_clip': display_val = f"{v / 10.0:.1f}"
                lbl.setText(str(display_val))
                sld.blockSignals(False)

        if self.preview_frame is not None:
            self.update_enhancement()

    def get_current_params(self):
        params = {}
        for k, (sld, _) in self.sliders.items():
            val = sld.value()
            if k == 'gamma': params[k] = val / 100.0
            elif k == 'clahe_clip': params[k] = val
            elif k == 'omega': params[k] = val
            else: params[k] = val
        return params

    def load_media(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Underwater Media", "",
            "All Supported (*.jpg *.jpeg *.png *.bmp *.tiff *.mp4 *.avi *.mov *.mkv);;Images (*.jpg *.png *.bmp);;Videos (*.mp4 *.avi *.mov)"
        )
        if not file_path: return

        self.stop_video()
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            self.load_video_file(file_path)
        else:
            self.load_image_file(file_path)

    def load_image_file(self, path):
        img_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None: return

        self.is_video = False
        self.original_frame = img
        self.preview_frame = self.downscale_preview(img)
        self.btn_play.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.btn_save.setEnabled(True)

        h, w = img.shape[:2]
        self.lbl_info.setText(f"Image: {w}x{h} px")
        self.status_bar.showMessage(f"Loaded image: {os.path.basename(path)}")
        self.update_enhancement()

    def load_video_file(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video source.")
            return

        self.is_video = True
        self.video_path = path
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interval = int(1000 / fps) if fps > 0 else 33

        self.btn_play.setEnabled(True)
        self.btn_snapshot.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_play.setText("⏸ Pause")

        ret, frame = self.cap.read()
        if ret:
            self.original_frame = frame
            self.preview_frame = self.downscale_preview(frame)
            self.update_enhancement()

        self.lbl_info.setText(f"Video: {w}x{h} @ {int(fps)} FPS")
        self.status_bar.showMessage(f"Playing video: {os.path.basename(path)}")
        self.timer.start(interval)

    def stop_video(self):
        if self.timer.isActive(): self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_video = False
        self.btn_play.setText("▶ Play")
        self.btn_play.setEnabled(False)
        self.btn_snapshot.setEnabled(False)

    def toggle_play_video(self):
        if not self.is_video or self.cap is None: return
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ Play")
            self.status_bar.showMessage("Video paused.")
        else:
            self.timer.start()
            self.btn_play.setText("⏸ Pause")
            self.status_bar.showMessage("Video playing...")

    def next_video_frame(self):
        if not self.is_video or self.cap is None: return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret: return

        self.original_frame = frame
        self.preview_frame = self.downscale_preview(frame)
        self.update_enhancement()

    def downscale_preview(self, img):
        if img is None: return None
        h, w = img.shape[:2]
        max_w = 1000
        if w > max_w:
            scale = max_w / float(w)
            return cv2.resize(img, (int(w * scale), int(h * scale)))
        return img.copy()

    def update_enhancement(self):
        if self.preview_frame is None: return
        params = self.get_current_params()
        self.processed_frame = ImageProcessor.process_pipeline(self.preview_frame, params)
        self.display_widget.set_images(self.preview_frame, self.processed_frame)

        # Update Live RGB Histogram
        hist_img = ImageProcessor.generate_histogram(self.processed_frame)
        if hist_img is not None:
            h, w, ch = hist_img.shape
            q_hist = QImage(hist_img.data, w, h, ch * w, QImage.Format_RGB888)
            self.lbl_hist.setPixmap(QPixmap.fromImage(q_hist))

    def on_view_mode_changed(self, index):
        mode = "side_by_side" if index == 0 else "split_slider"
        self.display_widget.set_mode(mode)

    def take_snapshot(self):
        if self.processed_frame is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "snapshot.jpg", "Images (*.jpg *.png)")
        if file_path:
            params = self.get_current_params()
            high_res_result = ImageProcessor.process_pipeline(self.original_frame, params)
            ext = os.path.splitext(file_path)[1] or ".jpg"
            success, enc = cv2.imencode(ext, high_res_result)
            if success:
                enc.tofile(file_path)
                self.status_bar.showMessage(f"Snapshot saved: {file_path}")
                QMessageBox.information(self, "Snapshot Saved", f"Frame snapshot saved to:\n{file_path}")

    def export_media(self):
        if self.is_video:
            self.export_video()
        else:
            self.export_image()

    def export_image(self):
        if self.original_frame is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Enhanced Image", "enhanced_photo.jpg", "Images (*.jpg *.png)")
        if path:
            self.status_bar.showMessage("Processing full resolution image...")
            params = self.get_current_params()
            final_res = ImageProcessor.process_pipeline(self.original_frame, params)
            ext = os.path.splitext(path)[1] or ".jpg"
            success, enc = cv2.imencode(ext, final_res)
            if success:
                enc.tofile(path)
                self.status_bar.showMessage(f"Saved: {path}")
                QMessageBox.information(self, "Success", f"Enhanced image saved successfully to:\n{path}")

    def export_video(self):
        if not self.video_path: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Enhanced Video", "enhanced_video.mp4", "Videos (*.mp4 *.avi)")
        if not path: return

        was_playing = self.timer.isActive()
        if was_playing: self.timer.stop()

        vcap = cv2.VideoCapture(self.video_path)
        w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if path.endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))

        params = self.get_current_params()
        prog = QProgressDialog("Rendering enhanced video frames...", "Cancel", 0, total_frames, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()

        idx = 0
        while True:
            ret, frame = vcap.read()
            if not ret or prog.wasCanceled(): break
            enhanced = ImageProcessor.process_pipeline(frame, params)
            out.write(enhanced)
            idx += 1
            prog.setValue(idx)
            QApplication.processEvents()

        vcap.release()
        out.release()
        prog.setValue(total_frames)

        if not prog.wasCanceled():
            QMessageBox.information(self, "Export Complete", f"Enhanced video rendered and saved to:\n{path}")
            self.status_bar.showMessage(f"Video exported: {path}")

        if was_playing: self.timer.start()

    def batch_process_folder(self):
        src_dir = QFileDialog.getExistingDirectory(self, "Select Folder Containing Underwater Images")
        if not src_dir: return

        dst_dir = os.path.join(src_dir, "AquaVision_Enhanced")
        os.makedirs(dst_dir, exist_ok=True)

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)]

        if not files:
            QMessageBox.warning(self, "No Images", "No supported image files found in the selected folder.")
            return

        params = self.get_current_params()
        prog = QProgressDialog("Batch processing images...", "Cancel", 0, len(files), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()

        for idx, f in enumerate(files):
            if prog.wasCanceled(): break
            in_p = os.path.join(src_dir, f)
            out_p = os.path.join(dst_dir, f)

            img_data = np.fromfile(in_p, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is not None:
                enhanced = ImageProcessor.process_pipeline(img, params)
                ext = os.path.splitext(out_p)[1] or ".jpg"
                s, enc = cv2.imencode(ext, enhanced)
                if s: enc.tofile(out_p)

            prog.setValue(idx + 1)
            QApplication.processEvents()

        if not prog.wasCanceled():
            QMessageBox.information(self, "Batch Complete", f"Processed {len(files)} images.\nSaved to: {dst_dir}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    studio = UnderwaterStudioApp()
    studio.show()
    sys.exit(app.exec_())
