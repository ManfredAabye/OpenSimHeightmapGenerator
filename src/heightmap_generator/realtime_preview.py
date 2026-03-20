import queue
import threading

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class Realtime3DPreview:
    def __init__(self, root, parent, title_prefix="3D Terrain"):
        self.root = root
        self.parent = parent
        self.title_prefix = title_prefix

        self.figure = Figure(figsize=(5.4, 4.2), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, self.parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_title("Echtzeit 3D")
        self.ax.set_xlabel("X (Meter)")
        self.ax.set_ylabel("Y (Meter)")
        self.ax.set_zlabel("Hoehe")
        self.canvas.draw_idle()

        self._input_queue = queue.Queue(maxsize=1)
        self._output_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._poll_output()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                payload = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:
                break

            height_data, title = payload
            rows = int(height_data.shape[0])
            cols = int(height_data.shape[1])
            step = max(1, rows // 100)

            x = np.arange(0, cols, step)
            y = np.arange(0, rows, step)
            X, Y = np.meshgrid(x, y)
            Z = height_data[::step, ::step]

            try:
                self._output_queue.put_nowait((X, Y, Z, title))
            except queue.Full:
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    pass
                self._output_queue.put_nowait((X, Y, Z, title))

    def _poll_output(self):
        if not self.parent.winfo_exists():
            return

        updated = False
        latest = None
        while True:
            try:
                latest = self._output_queue.get_nowait()
                updated = True
            except queue.Empty:
                break

        if updated and latest is not None:
            X, Y, Z, title = latest
            x_min = float(np.min(X))
            x_max = float(np.max(X))
            y_min = float(np.min(Y))
            y_max = float(np.max(Y))
            z_min = float(np.min(Z))
            z_max = float(np.max(Z))
            x_span = max(1e-6, x_max - x_min)
            y_span = max(1e-6, y_max - y_min)
            z_span = max(1e-6, z_max - z_min)

            self.ax.clear()
            self.ax.plot_surface(
                X,
                Y,
                Z,
                cmap="terrain",
                linewidth=0,
                antialiased=True,
                alpha=0.92,
            )
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_zlim(z_min, z_max)
            self.ax.set_box_aspect((x_span, y_span, z_span))
            self.ax.set_xlabel("X (Meter)")
            self.ax.set_ylabel("Y (Meter)")
            self.ax.set_zlabel("Hoehe (Meter)")
            self.ax.set_title(f"{self.title_prefix}: {title}")
            self.canvas.draw_idle()

        self.root.after(50, self._poll_output)

    def submit(self, height_data, title):
        item = (np.asarray(height_data, dtype=np.float32), title)
        try:
            self._input_queue.put_nowait(item)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            self._input_queue.put_nowait(item)

    def show_static(self, height_data, title):
        self.submit(height_data, title)

    def close(self):
        self._stop_event.set()
        try:
            self._input_queue.put_nowait(None)
        except queue.Full:
            pass
