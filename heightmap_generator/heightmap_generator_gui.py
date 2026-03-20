import os
import random
import threading
from datetime import datetime

import numpy as np
import tkinter as tk
from PIL import Image
from tkinter import filedialog, messagebox

import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, DISABLED, HORIZONTAL, LEFT, NORMAL, W, X, YES

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .models import GeneratorParams
from .realtime_preview import Realtime3DPreview
from .terrain_engine import TerrainEngine


class HeightmapGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenSim Heightmap Generator - Echtzeit 3D")
        self.root.geometry("1680x1280")
        self.root.minsize(1680, 1280)

        self.style = tb.Style(theme="darkly")
        self.engine = TerrainEngine()
        self.preview3d = None

        self.is_generating = False
        self._queued_live_refresh = False
        self._last_params = None
        self._realtime_after_id = None
        self._realtime_debounce_ms = 260

        self.current_image = None
        self.current_height_image = None
        self.height_gray_map = None
        self.preview_photo = None
        self.height_data = None
        self.slope_data = None
        self.roughness_data = None
        self.cost_data = None
        self.mixer_height_data = None
        self.mixer_source_path = None

        self.setup_ui()
        self._bind_realtime_triggers()

    def setup_ui(self):
        main_frame = tb.Frame(self.root, padding="20")
        main_frame.pack(fill=BOTH, expand=YES)
        main_frame.columnconfigure(0, weight=3, uniform="layout")
        main_frame.columnconfigure(1, weight=3, uniform="layout")
        main_frame.columnconfigure(2, weight=4, uniform="layout")
        main_frame.rowconfigure(1, weight=1)

        header = tb.Label(
            main_frame,
            text="OpenSim Heightmap Generator",
            font=("Helvetica", 18, "bold"),
            bootstyle="inverse-primary",
        )
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 16))

        generator_frame = tb.Labelframe(main_frame, text="Generator", padding="15")
        generator_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        middle_frame = tb.Frame(main_frame)
        middle_frame.grid(row=1, column=1, sticky="nsew", padx=5)
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.rowconfigure(3, weight=1)

        filter_frame = tb.Labelframe(middle_frame, text="Filter & Mixer", padding="15")
        filter_frame.grid(row=0, column=0, sticky="new")

        action_frame = tb.Labelframe(middle_frame, text="Aktionen", padding="12")
        action_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        status_frame = tb.Labelframe(middle_frame, text="Status", padding="12")
        status_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        profile_frame = tb.Labelframe(middle_frame, text="Hoehenprofil", padding="12")
        profile_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))

        preview_column = tb.Frame(main_frame)
        preview_column.grid(row=1, column=2, sticky="nsew", padx=(10, 0))
        preview_column.columnconfigure(0, weight=1)
        preview_column.rowconfigure(1, weight=1)
        preview_column.rowconfigure(3, weight=2)

        terrain_preview_frame = tb.Labelframe(preview_column, text="Generiertes Terrain", padding="12")
        terrain_preview_frame.grid(row=0, column=0, sticky="ew")

        preview_info_frame = tb.Labelframe(preview_column, text="Pixel-Info", padding="10")
        preview_info_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        preview3d_frame = tb.Labelframe(preview_column, text="3D Anzeige", padding="12")
        preview3d_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        self.preview3d = Realtime3DPreview(self.root, preview3d_frame, title_prefix="Echtzeit 3D")

        # ---- Standard HEX Referenz-Legende ----
        legend_frame = tb.Frame(generator_frame, bootstyle="secondary", padding=6)
        legend_frame.pack(fill=X, pady=(0, 10))
        tb.Label(legend_frame, text="Standard HEX Referenz:", font=("Helvetica", 9, "bold")).pack(anchor=W)
        ref_grid = tb.Frame(legend_frame)
        ref_grid.pack(fill=X, pady=(4, 0))
        for col, header in enumerate(("Name", "HEX", "Grau", "Hoehe")):
            tb.Label(ref_grid, text=header, font=("Helvetica", 8, "bold"), width=12).grid(row=0, column=col, sticky=W)
        for row, ref in enumerate(self.engine.TERRAIN_REFERENCES, start=1):
            tb.Label(ref_grid, text=ref["name"],              font=("Helvetica", 8), width=12).grid(row=row, column=0, sticky=W)
            tb.Label(ref_grid, text=ref["hex"],               font=("Courier", 8),  width=12).grid(row=row, column=1, sticky=W)
            tb.Label(ref_grid, text=str(ref["gray"]),         font=("Helvetica", 8), width=12).grid(row=row, column=2, sticky=W)
            tb.Label(ref_grid, text=f"{ref['height_m']:.1f} m", font=("Helvetica", 8), width=12).grid(row=row, column=3, sticky=W)

        tb.Label(generator_frame, text="Gelaendegroesse:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(0, 5))
        size_frame = tb.Frame(generator_frame)
        size_frame.pack(fill=X, pady=(0, 15))

        tb.Label(size_frame, text="Breite:").pack(side=LEFT, padx=(0, 5))
        self.width_var = tk.StringVar(value="256")
        tb.Spinbox(size_frame, from_=64, to=1024, textvariable=self.width_var, width=10, bootstyle="primary").pack(side=LEFT, padx=(0, 15))

        tb.Label(size_frame, text="Hoehe:").pack(side=LEFT, padx=(0, 5))
        self.height_var = tk.StringVar(value="256")
        tb.Spinbox(size_frame, from_=64, to=1024, textvariable=self.height_var, width=10, bootstyle="primary").pack(side=LEFT)

        preset_frame = tb.Frame(generator_frame)
        preset_frame.pack(fill=X, pady=(0, 10))
        tb.Label(preset_frame, text="Schnelleinstellung:").pack(side=LEFT, padx=(0, 8))
        for size in (256, 512, 768, 1024):
            tb.Button(
                preset_frame,
                text=f"{size}x{size}",
                command=lambda s=size: self.set_size_preset(s),
                bootstyle="outline-secondary",
                width=10,
            ).pack(side=LEFT, padx=(0, 6))

        self.terrain_type_var = tk.StringVar(value="round")
        tb.Label(generator_frame, text="Terrain-Typ:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))
        terrain_frame = tb.Frame(generator_frame)
        terrain_frame.pack(fill=X, pady=5)
        for text, value in [
            ("Rund", "round"),
            ("Quadratisch", "square"),
            ("Rechteckig", "rectangle"),
            ("Eliptisch", "ellipse"),
            ("Dreieckig", "triangle"),
            ("Kontinentale Insel", "continental_island"),
            ("Ozeanische Insel", "oceanic_island"),
            ("Atoll", "atoll"),
            ("Archipel", "archipelago"),
            ("Flussinsel", "river_island"),
            ("Dueneninsel", "dune_island"),
            ("Herzinsel", "heart_island"),
            ("Fussabdruck-Insel", "footprint_island"),
        ]:
            tb.Radiobutton(terrain_frame, text=text, variable=self.terrain_type_var, value=value, bootstyle="primary").pack(anchor=W, pady=2)
        tb.Label(
            generator_frame,
            text="Alle Formen werden mit abgerundeten Ecken erzeugt.",
            font=("Helvetica", 8),
            bootstyle="secondary",
        ).pack(anchor=W, pady=(0, 8))

        self.hill_extent_var = tk.DoubleVar(value=120.0)
        self.mountain_extent_var = tk.DoubleVar(value=180.0)
        self.octaves_var = tk.IntVar(value=3)
        self.persistence_var = tk.DoubleVar(value=0.3)
        self.base_height_var = tk.DoubleVar(value=0.0)
        self.hill_count_var = tk.IntVar(value=6)
        self.hill_height_var = tk.DoubleVar(value=5.0)
        self.mountain_count_var = tk.IntVar(value=2)
        self.mountain_height_var = tk.DoubleVar(value=30.0)
        self.gauss_var = tk.DoubleVar(value=2.0)

        tb.Label(generator_frame, text="Huegel:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(10, 5))
        self._add_scale_row(generator_frame, "Anzahl:", self.hill_count_var, 0, 20, "primary", "{:.0f}")
        self._add_scale_row(generator_frame, "Hoehe (m):", self.hill_height_var, 0.5, 30, "primary", "{:.1f}")
        self._add_scale_row(generator_frame, "Umfang:", self.hill_extent_var, 50, 300, "primary", "{:.1f}")

        tb.Label(generator_frame, text="Berge:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(12, 5))
        self._add_scale_row(generator_frame, "Anzahl:", self.mountain_count_var, 0, 12, "warning", "{:.0f}")
        self._add_scale_row(generator_frame, "Hoehe (m):", self.mountain_height_var, 1, 60, "warning", "{:.1f}")
        self._add_scale_row(generator_frame, "Umfang:", self.mountain_extent_var, 50, 300, "warning", "{:.1f}")

        tb.Label(generator_frame, text="Feindetails:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(12, 5))
        self._add_scale_row(generator_frame, "Detailstufe:", self.octaves_var, 1, 5, "primary", "{:.0f}")
        tb.Label(
            generator_frame,
            text="Steuert kleine Unebenheiten auf Huegeln und Bergen. Niedriger = ruhiger, hoeher = feiner strukturiert.",
            font=("Helvetica", 8),
            bootstyle="secondary",
            wraplength=840,
            justify=LEFT,
        ).pack(anchor=W, pady=(0, 6))

        self._add_scale_row(generator_frame, "Rauheit:", self.persistence_var, 0.1, 0.6, "primary", "{:.2f}")
        self._add_scale_row(generator_frame, "Glaetten:", self.gauss_var, 0.5, 5.0, "primary", "{:.1f}")
        tb.Label(
            generator_frame,
            text="Rauheit erzeugt Oberflaechenstruktur. Glaetten beruhigt diese wieder.",
            font=("Helvetica", 8),
            bootstyle="secondary",
        ).pack(anchor=W, pady=(0, 6))

        self._add_scale_row(generator_frame, "Offset ab NORMALNULL (m):", self.base_height_var, 0, 29, "primary", "{:.1f}")

        seed_frame = tb.Frame(generator_frame)
        seed_frame.pack(fill=X, pady=10)
        tb.Label(seed_frame, text="Seed:").pack(side=LEFT)
        self.seed_var = tk.IntVar(value=42)
        tb.Entry(seed_frame, textvariable=self.seed_var, width=15, bootstyle="primary").pack(side=LEFT, padx=(10, 5))
        tb.Button(seed_frame, text="Zufaellig", command=self.random_seed, bootstyle="secondary").pack(side=LEFT)
        tb.Label(
            generator_frame,
            text="Seed = Startwert fuer die Zufallsverteilung. Gleicher Seed + gleiche Einstellungen = identisches Terrain.",
            font=("Helvetica", 8),
            bootstyle="secondary",
            wraplength=840,
            justify=LEFT,
        ).pack(anchor=W, pady=(0, 10))

        tb.Label(filter_frame, text="Glaettungsfilter:", font=("Helvetica", 11, "bold")).pack(anchor=W, pady=(0, 10))
        self.median_var = tk.IntVar(value=1)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.auto_smooth_var = tk.BooleanVar(value=True)
        self.extra_smooth_var = tk.BooleanVar(value=False)
        self.realtime_3d_var = tk.BooleanVar(value=True)
        self.mixer_enabled_var = tk.BooleanVar(value=False)
        self.mixer_strength_var = tk.DoubleVar(value=0.5)
        self.mixer_mode_var = tk.StringVar(value="mix")
        self.cost_height_weight_var = tk.DoubleVar(value=0.15)
        self.cost_slope_weight_var = tk.DoubleVar(value=0.55)
        self.cost_rough_weight_var = tk.DoubleVar(value=0.30)

        self._add_scale_row(filter_frame, "Gauss-Filter:", self.gauss_var, 0.5, 5.0, "primary", "{:.1f}")

        median_frame = tb.Frame(filter_frame)
        median_frame.pack(fill=X, pady=5)
        tb.Label(median_frame, text="Median-Filter:").pack(side=LEFT)
        tb.Combobox(median_frame, textvariable=self.median_var, values=[1, 3, 5], width=10, bootstyle="primary", state="readonly").pack(side=LEFT, padx=(10, 10))

        self._add_scale_row(filter_frame, "Kontrast:", self.contrast_var, 0.5, 1.2, "primary", "{:.2f}")

        tb.Checkbutton(filter_frame, text="Automatische mehrstufige Glaettung", variable=self.auto_smooth_var, bootstyle="primary").pack(anchor=W, pady=6)
        tb.Checkbutton(filter_frame, text="Extreme Glaettung", variable=self.extra_smooth_var, bootstyle="secondary").pack(anchor=W, pady=6)
        tb.Checkbutton(filter_frame, text="Echtzeit-3D aktiv", variable=self.realtime_3d_var, bootstyle="success-round-toggle").pack(anchor=W, pady=8)

        mixer_box = tb.Labelframe(filter_frame, text="Terrain Mixer", padding=8)
        mixer_box.pack(fill=X, pady=(10, 0))
        tb.Checkbutton(mixer_box, text="Mixer aktiv", variable=self.mixer_enabled_var, bootstyle="warning-round-toggle").pack(anchor=W, pady=(0, 6))

        mode_frame = tb.Frame(mixer_box)
        mode_frame.pack(fill=X, pady=4)
        tb.Label(mode_frame, text="Blend-Modus:").pack(side=LEFT)
        tb.Combobox(
            mode_frame,
            textvariable=self.mixer_mode_var,
            values=["mix", "add", "multiply", "max", "min"],
            width=12,
            state="readonly",
            bootstyle="warning",
        ).pack(side=LEFT, padx=(10, 0))

        strength_frame = tb.Frame(mixer_box)
        strength_frame.pack(fill=X, pady=4)
        tb.Label(strength_frame, text="Staerke:").pack(side=LEFT)
        mixer_strength_scale = tb.Scale(
            strength_frame,
            from_=0.0,
            to=1.0,
            variable=self.mixer_strength_var,
            orient=HORIZONTAL,
            bootstyle="warning",
            length=260,
        )
        mixer_strength_scale.pack(side=LEFT, padx=(10, 10))
        self.mixer_strength_label = tb.Label(strength_frame, text="0.50")
        self.mixer_strength_label.pack(side=LEFT)
        mixer_strength_scale.config(command=lambda x: self.mixer_strength_label.config(text=f"{float(x):.2f}"))

        source_frame = tb.Frame(mixer_box)
        source_frame.pack(fill=X, pady=(6, 0))
        tb.Button(source_frame, text="Heightmap laden", command=self.load_mixer_heightmap, bootstyle="warning-outline").pack(side=LEFT)
        self.mixer_source_label = tb.Label(source_frame, text="Keine Mixer-Heightmap geladen", font=("Helvetica", 8))
        self.mixer_source_label.pack(side=LEFT, padx=(10, 0))

        cost_box = tb.Labelframe(filter_frame, text="Kostenkarte (Traversability)", padding=8)
        cost_box.pack(fill=X, pady=(10, 0))
        self._add_scale_row(cost_box, "Gewicht Hoehe:", self.cost_height_weight_var, 0.0, 1.0, "info", "{:.2f}")
        self._add_scale_row(cost_box, "Gewicht Steigung:", self.cost_slope_weight_var, 0.0, 1.0, "info", "{:.2f}")
        self._add_scale_row(cost_box, "Gewicht Rauheit:", self.cost_rough_weight_var, 0.0, 1.0, "info", "{:.2f}")
        preset_row = tb.Frame(cost_box)
        preset_row.pack(fill=X, pady=(6, 0))
        tb.Label(preset_row, text="Presets:").pack(side=LEFT)
        tb.Button(preset_row, text="Fahrbar", bootstyle="success-outline", width=11, command=lambda: self.apply_cost_preset("fahrbar")).pack(side=LEFT, padx=(8, 4))
        tb.Button(preset_row, text="Vorsichtig", bootstyle="warning-outline", width=11, command=lambda: self.apply_cost_preset("vorsichtig")).pack(side=LEFT, padx=4)
        tb.Button(preset_row, text="Sehr konservativ", bootstyle="danger-outline", width=15, command=lambda: self.apply_cost_preset("konservativ")).pack(side=LEFT, padx=4)
        tb.Label(cost_box, text="Hoeherer Kostenwert = schwieriger begehbar.", font=("Helvetica", 8), bootstyle="secondary").pack(anchor=W, pady=(2, 0))

        self.figure = Figure(figsize=(5.2, 3.0), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Entfernung (Meter)")
        self.ax.set_ylabel("Hoehe (Meter)")
        self.ax.set_title("Hoehenprofil")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.figure, profile_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=YES)

        self.progress = tb.Progressbar(status_frame, mode="indeterminate", bootstyle="success-striped")
        self.progress.pack(fill=X, pady=(0, 8))

        self.status_label = tb.Label(status_frame, text="Bereit", font=("Helvetica", 9))
        self.status_label.pack(anchor=W)

        button_frame = tb.Frame(action_frame)
        button_frame.pack(fill=X)
        self.generate_btn = tb.Button(button_frame, text="Terrain generieren", command=self.generate_heightmap, bootstyle="success", width=22)
        self.generate_btn.pack(fill=X, pady=(0, 8))

        self.save_btn = tb.Button(button_frame, text="Speichern", command=self.save_heightmap, bootstyle="primary", width=14, state=DISABLED)
        self.save_btn.pack(fill=X, pady=(0, 8))

        self.preview_btn = tb.Button(button_frame, text="3D Vorschau", command=self.show_3d_preview, bootstyle="info", width=14, state=DISABLED)
        self.preview_btn.pack(fill=X)

        self.export_layers_btn = tb.Button(button_frame, text="Layer Export", command=self.export_layers, bootstyle="secondary", state=DISABLED)
        self.export_layers_btn.pack(fill=X, pady=(8, 0))

        preview_container = tb.Frame(terrain_preview_frame, bootstyle="secondary", padding=6)
        preview_container.pack(fill=BOTH, expand=YES)
        self.preview_label = tb.Label(preview_container, text="Mini-Vorschau erscheint nach Generierung", anchor="center", width=34, padding=10)
        self.preview_label.pack(fill=BOTH, expand=YES)

        self.pixel_info_label = tb.Label(preview_info_frame, text="Pixel-Info: Vorschau generieren und mit der Maus ueber das Bild fahren", font=("Helvetica", 9), justify=LEFT, wraplength=420)
        self.pixel_info_label.pack(fill=X)

        layer_frame = tb.Frame(preview_info_frame)
        layer_frame.pack(fill=X, pady=(8, 0))
        tb.Label(layer_frame, text="Layer:").pack(side=LEFT)
        self.preview_layer_var = tk.StringVar(value="Hoehe")
        layer_combo = tb.Combobox(
            layer_frame,
            textvariable=self.preview_layer_var,
            values=["Hoehe", "Steigung", "Rauheit", "Kosten"],
            width=12,
            state="readonly",
            bootstyle="info",
        )
        layer_combo.pack(side=LEFT, padx=(8, 0))
        self.preview_layer_var.trace_add("write", self._on_preview_layer_changed)
        self.cost_height_weight_var.trace_add("write", self._on_cost_weights_changed)
        self.cost_slope_weight_var.trace_add("write", self._on_cost_weights_changed)
        self.cost_rough_weight_var.trace_add("write", self._on_cost_weights_changed)

        self.preview_label.bind("<Motion>", self._update_pixel_info)
        self.preview_label.bind("<Button-1>", self._update_pixel_info)
        self.preview_label.bind("<Leave>", self._clear_pixel_info)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _add_scale_row(self, parent, title, variable, from_, to, bootstyle, fmt):
        frame = tb.Frame(parent)
        frame.pack(fill=X, pady=5)
        tb.Label(frame, text=title).pack(side=LEFT)
        scale = tb.Scale(frame, from_=from_, to=to, variable=variable, orient=HORIZONTAL, bootstyle=bootstyle, length=300)
        scale.pack(side=LEFT, padx=(10, 10))
        label = tb.Label(frame, text=fmt.format(variable.get()))
        label.pack(side=LEFT)

        def _update(val):
            label.config(text=fmt.format(float(val)))

        scale.config(command=_update)

    def set_size_preset(self, size):
        value = str(int(size))
        self.width_var.set(value)
        self.height_var.set(value)

    def _bind_realtime_triggers(self):
        traces = [
            self.width_var,
            self.height_var,
            self.hill_extent_var,
            self.mountain_extent_var,
            self.hill_count_var,
            self.mountain_count_var,
            self.octaves_var,
            self.persistence_var,
            self.base_height_var,
            self.hill_height_var,
            self.mountain_height_var,
            self.seed_var,
            self.terrain_type_var,
            self.contrast_var,
            self.gauss_var,
            self.median_var,
            self.auto_smooth_var,
            self.extra_smooth_var,
            self.realtime_3d_var,
            self.mixer_enabled_var,
            self.mixer_strength_var,
            self.mixer_mode_var,
        ]
        for var in traces:
            var.trace_add("write", self._on_realtime_setting_changed)

    def _on_realtime_setting_changed(self, *_args):
        if not self.realtime_3d_var.get():
            return

        if self.is_generating:
            self._queued_live_refresh = True
            return

        self._schedule_realtime_refresh()

    def _schedule_realtime_refresh(self):
        if self._realtime_after_id is not None:
            try:
                self.root.after_cancel(self._realtime_after_id)
            except Exception:
                pass
        self._realtime_after_id = self.root.after(self._realtime_debounce_ms, self._run_realtime_refresh)

    def _run_realtime_refresh(self):
        self._realtime_after_id = None
        if not self.realtime_3d_var.get() or self.is_generating:
            if self.is_generating:
                self._queued_live_refresh = True
            return
        self.generate_heightmap(live_request=True)

    def _build_params(self):
        return GeneratorParams(
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            hill_extent=float(self.hill_extent_var.get()),
            mountain_extent=float(self.mountain_extent_var.get()),
            octaves=int(self.octaves_var.get()),
            persistence=float(self.persistence_var.get()),
            base_height=float(self.base_height_var.get()),
            hill_count=int(self.hill_count_var.get()),
            hill_height=float(self.hill_height_var.get()),
            mountain_count=int(self.mountain_count_var.get()),
            mountain_height=float(self.mountain_height_var.get()),
            seed=int(self.seed_var.get()),
            terrain_type=self.terrain_type_var.get(),
            contrast=float(self.contrast_var.get()),
            gauss_sigma=float(self.gauss_var.get()),
            median_size=int(self.median_var.get()),
            auto_smooth=bool(self.auto_smooth_var.get()),
            extra_smooth=bool(self.extra_smooth_var.get()),
        )

    def _on_preview_layer_changed(self, *_args):
        self._refresh_preview_layer()

    def _on_cost_weights_changed(self, *_args):
        if self.height_data is None:
            return
        self._compute_cost_layer()
        self._refresh_preview_layer()

    def apply_cost_preset(self, preset_name: str):
        if preset_name == "fahrbar":
            weights = (0.10, 0.35, 0.55)
        elif preset_name == "vorsichtig":
            weights = (0.15, 0.55, 0.30)
        else:
            weights = (0.20, 0.70, 0.10)

        self.cost_height_weight_var.set(weights[0])
        self.cost_slope_weight_var.set(weights[1])
        self.cost_rough_weight_var.set(weights[2])
        self.status_label.config(text=f"Kosten-Preset aktiv: {preset_name}")

    def _compute_cost_layer(self):
        if self.height_data is None or self.slope_data is None or self.roughness_data is None:
            self.cost_data = None
            return

        h_min = float(np.min(self.height_data))
        h_max = float(np.max(self.height_data))
        h_span = max(1e-6, h_max - h_min)
        height_norm = np.clip((self.height_data - h_min) / h_span, 0.0, 1.0)

        slope_norm = np.clip(self.slope_data / 45.0, 0.0, 1.0)
        rough_ref = float(np.percentile(self.roughness_data, 95.0))
        rough_norm = np.clip(self.roughness_data / max(1e-6, rough_ref), 0.0, 1.0)

        w_h = float(max(0.0, self.cost_height_weight_var.get()))
        w_s = float(max(0.0, self.cost_slope_weight_var.get()))
        w_r = float(max(0.0, self.cost_rough_weight_var.get()))
        w_sum = max(1e-6, w_h + w_s + w_r)

        self.cost_data = (w_h * height_norm + w_s * slope_norm + w_r * rough_norm) / w_sum

    def _refresh_preview_layer(self):
        if self.height_data is None:
            return

        selected_layer = self.preview_layer_var.get() if hasattr(self, "preview_layer_var") else "Hoehe"
        if selected_layer == "Steigung" and self.slope_data is not None:
            layer_gray = self.engine.layer_to_gray(self.slope_data)
        elif selected_layer == "Rauheit" and self.roughness_data is not None:
            layer_gray = self.engine.layer_to_gray(self.roughness_data)
        elif selected_layer == "Kosten" and self.cost_data is not None:
            layer_gray = np.rint(np.clip(self.cost_data, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            layer_gray = self.height_gray_map

        if layer_gray is None:
            return

        self.current_image = Image.fromarray(layer_gray, mode="L")
        preview_image = self.current_image.copy()
        preview_image.thumbnail((440, 440))

        from PIL import ImageTk

        self.preview_photo = ImageTk.PhotoImage(preview_image)
        self.preview_label.config(image=self.preview_photo, text="")

    def random_seed(self):
        self.seed_var.set(random.randint(1, 10000))

    def load_mixer_heightmap(self):
        file_path = filedialog.askopenfilename(
            title="Mixer Heightmap laden",
            filetypes=[("Bilddateien", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"), ("Alle Dateien", "*.*")],
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert("L")
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.BILINEAR)
            gray_data = np.array(img, dtype=np.uint8)
            self.mixer_height_data = self.engine.gray_to_height_array(gray_data)
            self.mixer_source_path = file_path
            self.mixer_source_label.config(text=os.path.basename(file_path))
            self.status_label.config(text=f"Mixer geladen: {os.path.basename(file_path)}")

            if self.realtime_3d_var.get() and not self.is_generating:
                self._schedule_realtime_refresh()
        except Exception as err:
            messagebox.showerror("Mixer-Fehler", f"Konnte Mixer-Heightmap nicht laden:\n{err}")

    def _apply_mixer(self, base_height_data: np.ndarray) -> np.ndarray:
        if not self.mixer_enabled_var.get() or self.mixer_height_data is None:
            return base_height_data

        mix = np.asarray(self.mixer_height_data, dtype=np.float32)
        if mix.shape != base_height_data.shape:
            return base_height_data

        strength = float(np.clip(self.mixer_strength_var.get(), 0.0, 1.0))
        mode = self.mixer_mode_var.get()

        # Normalisierte Darstellung ab NORMALNULL (0 m) bis MAX_HEIGHT.
        base_n = np.clip(base_height_data / self.engine.MAX_HEIGHT, 0.0, 1.0)
        mix_n = np.clip(mix / self.engine.MAX_HEIGHT, 0.0, 1.0)

        if mode == "add":
            out_n = np.clip(base_n + mix_n * strength, 0.0, 1.0)
        elif mode == "multiply":
            mult = (1.0 - strength) + strength * mix_n
            out_n = np.clip(base_n * mult, 0.0, 1.0)
        elif mode == "max":
            out_n = np.maximum(base_n, mix_n * strength)
        elif mode == "min":
            out_n = np.minimum(base_n, mix_n)
            out_n = (1.0 - strength) * base_n + strength * out_n
        else:
            out_n = (1.0 - strength) * base_n + strength * mix_n

        return np.clip(out_n * self.engine.MAX_HEIGHT, self.engine.MIN_HEIGHT, self.engine.MAX_HEIGHT)

    def generate_heightmap(self, live_request=False):
        if self.is_generating:
            if live_request:
                self._queued_live_refresh = True
            return

        try:
            params = self._build_params()
        except Exception as err:
            messagebox.showerror("Eingabefehler", str(err))
            return

        self._last_params = params
        self.is_generating = True
        self.generate_btn.config(state=DISABLED)
        self.save_btn.config(state=DISABLED)
        self.preview_btn.config(state=DISABLED)
        self.progress.start(10)
        self.status_label.config(text="Generiere Terrain...")

        thread = threading.Thread(target=self._generate_thread, args=(params,), daemon=True)
        thread.start()

    def _generate_thread(self, params: GeneratorParams):
        try:
            height_data, gray_map, analysis_layers = self.engine.generate(params)
            height_data = self._apply_mixer(height_data)
            gray_map = self.engine.height_to_gray_array(height_data)
            # Nur bei aktivem Mixer neu berechnen, sonst die bereits berechneten Layer nutzen.
            if self.mixer_enabled_var.get() and self.mixer_height_data is not None:
                analysis_layers = self.engine.compute_analysis_layers(height_data)
            self.height_data = height_data
            self.height_gray_map = gray_map
            self.slope_data = analysis_layers.get("slope")
            self.roughness_data = analysis_layers.get("roughness")
            self._compute_cost_layer()
            self.current_height_image = Image.fromarray(gray_map, mode="L")

            self.root.after(0, self._update_gui_after_generation, params)
        except Exception as err:
            self.root.after(0, self._show_error, str(err))

    def _update_gui_after_generation(self, params: GeneratorParams):
        self._refresh_preview_layer()
        self.pixel_info_label.config(text="Pixel-Info: Mit Maus ueber die Vorschau fahren")

        self.update_height_profile()

        self.generate_btn.config(state=NORMAL)
        self.save_btn.config(state=NORMAL)
        self.preview_btn.config(state=NORMAL)
        self.export_layers_btn.config(state=NORMAL)
        self.progress.stop()

        if self.height_data is not None:
            min_h = float(np.min(self.height_data))
            max_h = float(np.max(self.height_data))
            self.status_label.config(text=f"Terrain generiert | Hoehen: {min_h:.1f}m bis {max_h:.1f}m")

            if self.realtime_3d_var.get() and self.preview3d is not None:
                title = f"{params.width}x{params.height} | {params.terrain_type} | Seed {params.seed}"
                self.preview3d.submit(self.height_data, title)

        self.is_generating = False

        if self._queued_live_refresh and self.realtime_3d_var.get():
            self._queued_live_refresh = False
            self._schedule_realtime_refresh()

    def _show_error(self, error_msg):
        self.generate_btn.config(state=NORMAL)
        self.progress.stop()
        self.status_label.config(text=f"Fehler: {error_msg}")
        self.is_generating = False
        messagebox.showerror("Fehler", f"Bei der Generierung ist ein Fehler aufgetreten:\n{error_msg}")

    def update_height_profile(self):
        if self.height_data is None:
            return

        self.ax.clear()
        profile = np.mean(self.height_data, axis=0)
        distance = np.arange(len(profile))

        self.ax.plot(distance, profile, "g-", linewidth=2, alpha=0.8)
        self.ax.fill_between(distance, self.engine.MIN_HEIGHT, profile, alpha=0.3, color="green")

        min_h = float(np.min(self.height_data))
        max_h = float(np.max(self.height_data))
        mean_h = float(np.mean(self.height_data))

        self.ax.set_xlabel("Entfernung (Meter)")
        self.ax.set_ylabel("Hoehe (Meter)")
        self.ax.set_title(f"Hoehenprofil | Min: {min_h:.1f}m | Max: {max_h:.1f}m | Mittel: {mean_h:.1f}m")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(self.engine.MIN_HEIGHT - 10, self.engine.MAX_HEIGHT + 10)
        self.canvas.draw()

    def save_heightmap(self):
        if self.current_height_image is None or self.height_data is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"homogen_{self.width_var.get()}x{self.height_var.get()}_{timestamp}"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Bilder", "*.png"), ("RAW Daten", "*.raw"), ("CSV Hoehendaten", "*.csv"), ("Alle Dateien", "*.*")],
            initialfile=default_name,
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".raw"):
                with open(file_path, "wb") as file_handle:
                    file_handle.write(self.height_data.astype(np.float32).tobytes())
            elif file_path.endswith(".csv"):
                np.savetxt(file_path, self.height_data, delimiter=",", fmt="%.2f")
            else:
                self.current_height_image.save(file_path)
                meta_path = file_path.rsplit(".", 1)[0] + "_info.txt"
                with open(meta_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write("Homogener Heightmap Generator\n")
                    file_handle.write("============================\n")
                    file_handle.write(f"Terrain-Typ: {self.terrain_type_var.get()}\n")
                    file_handle.write(f"Groesse: {self.width_var.get()} x {self.height_var.get()} Meter\n")
                    file_handle.write(f"Hoehenbereich: {self.engine.MIN_HEIGHT}m bis {self.engine.MAX_HEIGHT}m\n")
                    file_handle.write(f"Huegelhoehe: {self.hill_height_var.get():.2f}m\n")
                    file_handle.write(f"Berghoehe: {self.mountain_height_var.get():.2f}m\n")
                    file_handle.write(f"Minimale Hoehe: {np.min(self.height_data):.2f}m\n")
                    file_handle.write(f"Maximale Hoehe: {np.max(self.height_data):.2f}m\n")
                    file_handle.write(f"Seed: {self.seed_var.get()}\n")

            self.status_label.config(text=f"Gespeichert: {os.path.basename(file_path)}")
        except Exception as err:
            messagebox.showerror("Fehler", f"Konnte Datei nicht speichern:\n{err}")

    def show_3d_preview(self):
        if self.height_data is None or self.preview3d is None:
            return

        title = f"{self.width_var.get()}x{self.height_var.get()} | {self.terrain_type_var.get()}"
        self.preview3d.show_static(self.height_data, title)

    def _update_pixel_info(self, event):
        if self.current_image is None or self.preview_photo is None or self.height_gray_map is None or self.height_data is None:
            return

        preview_w = int(self.preview_photo.width())
        preview_h = int(self.preview_photo.height())
        if preview_w <= 0 or preview_h <= 0:
            return

        widget_w = int(self.preview_label.winfo_width())
        widget_h = int(self.preview_label.winfo_height())
        offset_x = max(0, (widget_w - preview_w) // 2)
        offset_y = max(0, (widget_h - preview_h) // 2)

        local_x = int(event.x) - offset_x
        local_y = int(event.y) - offset_y
        if local_x < 0 or local_y < 0 or local_x >= preview_w or local_y >= preview_h:
            return

        img_w, img_h = self.current_image.size
        src_x = min(img_w - 1, max(0, int(local_x * img_w / preview_w)))
        src_y = min(img_h - 1, max(0, int(local_y * img_h / preview_h)))

        gray = int(self.height_gray_map[src_y, src_x])
        height_m = float(self.height_data[src_y, src_x])
        slope_m = float(self.slope_data[src_y, src_x]) if self.slope_data is not None else 0.0
        rough_m = float(self.roughness_data[src_y, src_x]) if self.roughness_data is not None else 0.0
        cost_v = float(self.cost_data[src_y, src_x]) if self.cost_data is not None else 0.0
        hex_triplet = f"{gray:02X}{gray:02X}{gray:02X}"

        self.pixel_info_label.config(
            text=f"Pixel-Info: HEX #{hex_triplet} | Graustufe {gray} | Hoehe {height_m:.2f} m | Steigung {slope_m:.2f} deg | Rauheit {rough_m:.2f} m | Kosten {cost_v:.2f} | Position {src_x},{src_y}"
        )

    def _clear_pixel_info(self, _event):
        self.pixel_info_label.config(text="Pixel-Info: Maus ausserhalb der Vorschau")

    def export_layers(self):
        if self.height_data is None or self.slope_data is None or self.roughness_data is None:
            return

        self._compute_cost_layer()
        if self.cost_data is None:
            return

        export_dir = filedialog.askdirectory(title="Export-Ordner fuer Layer auswaehlen")
        if not export_dir:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"layers_{self.width_var.get()}x{self.height_var.get()}_{timestamp}"

            height_gray = self.height_gray_map
            if height_gray is None:
                height_gray = self.engine.height_to_gray_array(self.height_data)
            slope_gray = self.engine.layer_to_gray(self.slope_data)
            rough_gray = self.engine.layer_to_gray(self.roughness_data)
            cost_gray = np.rint(np.clip(self.cost_data, 0.0, 1.0) * 255.0).astype(np.uint8)

            Image.fromarray(height_gray, mode="L").save(os.path.join(export_dir, f"{prefix}_height.png"))
            Image.fromarray(slope_gray, mode="L").save(os.path.join(export_dir, f"{prefix}_slope.png"))
            Image.fromarray(rough_gray, mode="L").save(os.path.join(export_dir, f"{prefix}_roughness.png"))
            Image.fromarray(cost_gray, mode="L").save(os.path.join(export_dir, f"{prefix}_cost.png"))

            np.savetxt(os.path.join(export_dir, f"{prefix}_height.csv"), self.height_data, delimiter=",", fmt="%.4f")
            np.savetxt(os.path.join(export_dir, f"{prefix}_slope.csv"), self.slope_data, delimiter=",", fmt="%.4f")
            np.savetxt(os.path.join(export_dir, f"{prefix}_roughness.csv"), self.roughness_data, delimiter=",", fmt="%.4f")
            np.savetxt(os.path.join(export_dir, f"{prefix}_cost.csv"), self.cost_data, delimiter=",", fmt="%.4f")

            self.height_data.astype(np.float32).tofile(os.path.join(export_dir, f"{prefix}_height.raw"))
            self.slope_data.astype(np.float32).tofile(os.path.join(export_dir, f"{prefix}_slope.raw"))
            self.roughness_data.astype(np.float32).tofile(os.path.join(export_dir, f"{prefix}_roughness.raw"))
            self.cost_data.astype(np.float32).tofile(os.path.join(export_dir, f"{prefix}_cost.raw"))

            self.status_label.config(text=f"Layer exportiert: {os.path.basename(export_dir)}")
        except Exception as err:
            messagebox.showerror("Export-Fehler", f"Layer konnten nicht exportiert werden:\n{err}")

    def _on_close(self):
        if self._realtime_after_id is not None:
            try:
                self.root.after_cancel(self._realtime_after_id)
            except Exception:
                pass
        if self.preview3d is not None:
            self.preview3d.close()
        self.root.destroy()
