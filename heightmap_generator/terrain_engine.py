import noise
import numpy as np
from matplotlib.path import Path
from scipy import ndimage
from scipy import signal

from .models import GeneratorParams


class TerrainEngine:
    MIN_HEIGHT = -10.0
    MAX_HEIGHT = 30.0

    # Standard HEX Referenzwerte (Graustufen exakt aus den Hex-Angaben)
    # NORMALNULL=#282828  gray=40  0 m
    # LAND      =#292929  gray=41  1 m
    # HUEGEL    =#323232  gray=50  9.4 m  (linear aus Kalibrierung)
    # BERG      =#464646  gray=70  29 m   (linear aus Kalibrierung)
    TERRAIN_REFERENCES = [
        {"name": "NORMALNULL", "hex": "#282828", "gray": 40, "height_m":  0.0},
        {"name": "LAND",       "hex": "#292929", "gray": 41, "height_m":  1.0},
        {"name": "HUEGEL",     "hex": "#323232", "gray": 50, "height_m":  9.4},
        {"name": "BERG",       "hex": "#464646", "gray": 70, "height_m": 29.0},
    ]

    def __init__(self):
        # Kalibrierungskurve – beinhaltet jetzt alle vier Standard-Referenzpunkte
        # explizit, damit interp exakt auf den definierten Werten landet.
        # Bei -10 m liegt der Grauwert linear bei 36  (90/100 * 40 = 36)
        self.height_points_m = np.array([-10.0, 0.0, 1.0, 9.4, 15.0, 29.0, 30.0], dtype=float)
        self.gray_points      = np.array([ 36.0, 40.0, 41.0, 50.0, 56.0, 70.0, 71.0], dtype=float)

    def height_to_gray_array(self, heights):
        clipped = np.clip(heights, self.MIN_HEIGHT, self.MAX_HEIGHT)
        gray = np.interp(clipped, self.height_points_m, self.gray_points)
        return np.rint(gray).astype(np.uint8)

    def gray_to_height_array(self, gray_values):
        gray = np.clip(gray_values.astype(float), 0.0, 255.0)
        return np.interp(gray, self.gray_points, self.height_points_m)

    def compute_analysis_layers(self, height_data: np.ndarray):
        grad_y, grad_x = np.gradient(height_data)
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad).astype(np.float32)

        smooth = ndimage.gaussian_filter(height_data, sigma=2.0)
        rough_base = (height_data - smooth) ** 2
        roughness = np.sqrt(ndimage.gaussian_filter(rough_base, sigma=2.0)).astype(np.float32)

        return {
            "slope": slope_deg,
            "roughness": roughness,
        }

    @staticmethod
    def layer_to_gray(layer_data: np.ndarray, min_percentile=2.0, max_percentile=98.0) -> np.ndarray:
        low = float(np.percentile(layer_data, min_percentile))
        high = float(np.percentile(layer_data, max_percentile))
        if high <= low:
            return np.zeros_like(layer_data, dtype=np.uint8)
        norm = np.clip((layer_data - low) / (high - low), 0.0, 1.0)
        return np.rint(norm * 255.0).astype(np.uint8)

    def generate_homogeneous_terrain(self, width, height, extent, octaves, persistence, seed):
        x = np.linspace(0, width / extent, width)
        y = np.linspace(0, height / extent, height)
        X, Y = np.meshgrid(x, y)

        terrain = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                n1 = noise.pnoise2(
                    X[i, j] + seed,
                    Y[i, j] + seed,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=seed,
                )
                n2 = noise.pnoise2(
                    X[i, j] * 0.5 + seed * 2,
                    Y[i, j] * 0.5 + seed * 2,
                    octaves=max(1, octaves - 1),
                    persistence=persistence * 0.8,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=seed + 100,
                )
                n3 = noise.pnoise2(
                    X[i, j] * 0.2 + seed * 3,
                    Y[i, j] * 0.2 + seed * 3,
                    octaves=2,
                    persistence=0.3,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=seed + 200,
                )
                terrain[i, j] = n1 * 0.6 + n2 * 0.3 + n3 * 0.1

        return terrain

    def generate_wavy_terrain(self, width, height):
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        waves = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2 * X) * np.cos(2 * Y)
        return waves / 2.0

    def generate_island(self, width, height, extent, seed):
        terrain = self.generate_homogeneous_terrain(width, height, extent, 3, 0.3, seed)

        center_x, center_y = width / 2.0, height / 2.0
        max_dist = np.sqrt(center_x**2 + center_y**2)

        for i in range(height):
            for j in range(width):
                dist = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
                mask = 1.0 - (dist / max_dist) ** 1.5
                terrain[i, j] *= max(0.0, mask)

        return terrain

    def generate_mountain_range(self, width, height, extent, seed):
        terrain = self.generate_homogeneous_terrain(width, height, extent * 0.7, 4, 0.4, seed)
        center = height // 2
        for i in range(height):
            factor = 1.0 - abs(i - center) / (height / 2) * 0.7
            terrain[i, :] *= factor
        return terrain

    def apply_smoothing(self, data, params: GeneratorParams):
        if params.gauss_sigma > 0:
            data = ndimage.gaussian_filter(data, sigma=params.gauss_sigma)

        if params.median_size > 1:
            data = ndimage.median_filter(data, size=params.median_size)

        if params.auto_smooth:
            data = ndimage.gaussian_filter(data, sigma=1.5)
            data = ndimage.gaussian_filter(data, sigma=1.0)
            data = ndimage.gaussian_filter(data, sigma=0.5)

        if params.extra_smooth:
            data = ndimage.gaussian_filter(data, sigma=3.0)
            data = ndimage.gaussian_filter(data, sigma=2.0)
            data = ndimage.uniform_filter(data, size=5)

        return data

    def _apply_edge_fade(self, height_data: np.ndarray, amplitude: float) -> np.ndarray:
        """Fade map edges to NORMALNULL with a border proportional to amplitude.

        The border width equals max(8 % of map size, amplitude * SLOPE_RATIO).
        SLOPE_RATIO = 2.0 px/m  (approx. 27-degree natural slope).
        This guarantees that any feature in the interior has its complete base
        visible and is never cut off at the tile edge.
        Smoothstep is used for a natural-looking transition.
        """
        SLOPE_RATIO = 2.0
        rows, cols = height_data.shape
        min_border = max(4, int(min(rows, cols) * 0.08))
        amp_border = int(amplitude * SLOPE_RATIO)
        border = max(min_border, amp_border)
        border = min(border, max(1, min(rows, cols) // 2))

        y = np.arange(rows, dtype=np.float32)
        x = np.arange(cols, dtype=np.float32)
        y_dist = np.minimum(y, (rows - 1.0) - y)[:, None]
        x_dist = np.minimum(x, (cols - 1.0) - x)[None, :]
        edge_dist = np.minimum(y_dist, x_dist)

        t = np.clip(edge_dist / float(border), 0.0, 1.0)
        blend = t * t * (3.0 - 2.0 * t)   # smoothstep
        return height_data * blend

    def _enforce_natural_base(self, height_data: np.ndarray, amplitude: float) -> np.ndarray:
        """Ensure features have a footprint proportional to their height.

        Applies a Gaussian with sigma = amplitude / 8.0 (minimum 1.0).
        Taller features get more lateral spreading, so a 30 m mountain
        automatically has a wider base than a 5 m hill.
        This runs AFTER user-defined smoothing so it does not double-apply
        the user's own Gauss filter.
        """
        if amplitude <= 0.0:
            return height_data
        sigma = max(1.0, amplitude / 8.0)
        return ndimage.gaussian_filter(height_data, sigma=sigma)

    def _create_shape_profile(self, shape_name: str, extent: float, rng: np.random.Generator) -> np.ndarray:
        min_extent = max(12.0, float(extent))
        base_radius = max(6.0, min_extent / 2.0)
        rotation = float(rng.uniform(0.0, np.pi))
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        if shape_name == "round":
            radius_x = base_radius
            radius_y = base_radius
            power = 2.0
        elif shape_name == "square":
            radius_x = base_radius
            radius_y = base_radius
            power = 5.5
        elif shape_name == "rectangle":
            ratio = float(rng.uniform(1.3, 2.0))
            radius_x = base_radius * ratio
            radius_y = base_radius / ratio * 1.15
            power = 5.5
        elif shape_name == "ellipse":
            ratio = float(rng.uniform(1.2, 1.9))
            radius_x = base_radius * ratio
            radius_y = base_radius / ratio * 1.25
            power = 2.0
        else:
            radius_x = base_radius * 1.15
            radius_y = base_radius * 1.15
            power = None

        pad = int(np.ceil(max(radius_x, radius_y) * 1.8))
        grid = np.arange(-pad, pad + 1, dtype=np.float32)
        X, Y = np.meshgrid(grid, grid)
        xr = X * cos_r + Y * sin_r
        yr = -X * sin_r + Y * cos_r

        def superellipse(xv, yv, rx, ry, pwr):
            r = (np.abs(xv / rx) ** pwr + np.abs(yv / ry) ** pwr) ** (1.0 / pwr)
            return np.clip(1.0 - r, 0.0, 1.0).astype(np.float32)

        if shape_name == "continental_island":
            core = superellipse(xr, yr, radius_x * 1.1, radius_y * 0.9, 2.4)
            shelf = superellipse(xr, yr, radius_x * 1.6, radius_y * 1.25, 2.6) * 0.45
            profile = np.maximum(core, shelf)
        elif shape_name == "oceanic_island":
            volcano = np.clip(1.0 - np.sqrt((xr / (radius_x * 0.95)) ** 2 + (yr / (radius_y * 0.95)) ** 2), 0.0, 1.0)
            cone = volcano ** 2.3
            flank = np.clip(1.0 - np.sqrt((xr / (radius_x * 1.6)) ** 2 + (yr / (radius_y * 1.6)) ** 2), 0.0, 1.0) * 0.35
            profile = np.maximum(cone, flank).astype(np.float32)
        elif shape_name == "atoll":
            outer = np.clip(1.0 - np.sqrt((xr / (radius_x * 1.15)) ** 2 + (yr / (radius_y * 1.15)) ** 2), 0.0, 1.0)
            inner = np.clip(1.0 - np.sqrt((xr / (radius_x * 0.58)) ** 2 + (yr / (radius_y * 0.58)) ** 2), 0.0, 1.0)
            ring = np.clip(outer - inner * 1.6, 0.0, 1.0)
            profile = (ring ** 1.1).astype(np.float32)
        elif shape_name == "archipelago":
            profile = np.zeros_like(xr, dtype=np.float32)
            island_count = int(rng.integers(3, 8))
            for _ in range(island_count):
                ox = float(rng.uniform(-radius_x * 0.9, radius_x * 0.9))
                oy = float(rng.uniform(-radius_y * 0.9, radius_y * 0.9))
                rx = float(rng.uniform(radius_x * 0.25, radius_x * 0.45))
                ry = float(rng.uniform(radius_y * 0.25, radius_y * 0.45))
                pwr = float(rng.uniform(1.8, 3.0))
                part = superellipse(xr - ox, yr - oy, rx, ry, pwr)
                profile = np.maximum(profile, part)
        elif shape_name == "river_island":
            main = superellipse(xr, yr, radius_x * 1.8, radius_y * 0.55, 2.6)
            taper_left = np.clip((xr + radius_x * 1.6) / (radius_x * 0.9), 0.0, 1.0)
            taper_right = np.clip((radius_x * 1.6 - xr) / (radius_x * 0.9), 0.0, 1.0)
            profile = (main * taper_left * taper_right).astype(np.float32)
        elif shape_name == "dune_island":
            ridge = superellipse(xr, yr, radius_x * 1.9, radius_y * 0.48, 2.2)
            crest = np.exp(-((yr / (radius_y * 0.28)) ** 2))
            profile = np.clip(ridge * (0.55 + 0.45 * crest), 0.0, 1.0).astype(np.float32)
        elif shape_name == "heart_island":
            nx = xr / (radius_x * 1.55)
            ny = yr / (radius_y * 1.55)
            heart = (nx**2 + ny**2 - 1.0) ** 3 - nx**2 * ny**3
            profile = np.clip(1.0 - np.maximum(heart, 0.0) * 4.5, 0.0, 1.0).astype(np.float32)
        elif shape_name == "footprint_island":
            heel = superellipse(xr + radius_x * 0.25, yr + radius_y * 0.20, radius_x * 0.65, radius_y * 0.90, 2.4)
            ball = superellipse(xr + radius_x * 0.18, yr - radius_y * 0.55, radius_x * 0.58, radius_y * 0.45, 2.4)
            toes = np.zeros_like(xr, dtype=np.float32)
            toe_offsets = [(-0.60, -0.95), (-0.30, -1.05), (0.00, -1.08), (0.30, -1.00), (0.55, -0.88)]
            for ox, oy in toe_offsets:
                toes = np.maximum(
                    toes,
                    superellipse(
                        xr - ox * radius_x,
                        yr - oy * radius_y,
                        radius_x * 0.16,
                        radius_y * 0.17,
                        2.0,
                    ),
                )
            profile = np.clip(np.maximum(np.maximum(heel, ball), toes), 0.0, 1.0).astype(np.float32)

        elif shape_name == "triangle":
            nx = xr / radius_x
            ny = yr / radius_y
            points = np.column_stack((nx.ravel(), ny.ravel()))
            triangle = Path([(0.0, -1.0), (-0.92, 0.70), (0.92, 0.70)])
            profile = triangle.contains_points(points).reshape(nx.shape).astype(np.float32)
        else:
            assert power is not None
            profile = superellipse(xr, yr, radius_x, radius_y, power)

        round_sigma = max(1.0, max(radius_x, radius_y) * 0.08)
        profile = ndimage.gaussian_filter(profile, sigma=round_sigma)
        max_profile = float(np.max(profile))
        if max_profile > 0.0:
            profile = profile / max_profile
        return np.clip(profile, 0.0, 1.0) ** 1.6

    def _place_features(self, terrain: np.ndarray, count: int, height_value: float, extent: float, shape_name: str, rng: np.random.Generator):
        rows, cols = terrain.shape
        count = max(0, int(count))
        if count <= 0 or height_value <= 0.0:
            return

        effective_extent = max(float(extent), float(height_value) * 4.0)
        stamp = self._create_shape_profile(shape_name, effective_extent, rng) * float(height_value)
        stamp_h, stamp_w = stamp.shape
        half_h = stamp_h // 2
        half_w = stamp_w // 2
        margin_y = min(max(half_h + 2, int(height_value * 2.0)), max(1, rows // 2))
        margin_x = min(max(half_w + 2, int(height_value * 2.0)), max(1, cols // 2))

        if cols > margin_x * 2:
            centers_x = rng.integers(margin_x, cols - margin_x, size=count)
        else:
            centers_x = np.full(count, cols // 2, dtype=np.int32)
        if rows > margin_y * 2:
            centers_y = rng.integers(margin_y, rows - margin_y, size=count)
        else:
            centers_y = np.full(count, rows // 2, dtype=np.int32)

        impulses = np.zeros_like(terrain, dtype=np.float32)
        np.add.at(impulses, (centers_y, centers_x), 1.0)

        # C-optimierte Faltung statt Python-Schleife ueber jedes Feature.
        # Bei grossen Stamps ist FFT deutlich schneller als direkte Faltung.
        if stamp.size >= 64 * 64:
            conv = signal.fftconvolve(impulses, stamp, mode="same")
            terrain += conv.astype(np.float32, copy=False)
        else:
            terrain += ndimage.convolve(impulses, stamp, mode="constant", cval=0.0)

    def generate_feature_terrain(self, params: GeneratorParams):
        rng = np.random.default_rng(int(params.seed))
        terrain = np.zeros((params.height, params.width), dtype=np.float32)

        self._place_features(
            terrain,
            params.hill_count,
            params.hill_height,
            params.hill_extent,
            params.terrain_type,
            rng,
        )
        self._place_features(
            terrain,
            params.mountain_count,
            params.mountain_height,
            params.mountain_extent,
            params.terrain_type,
            rng,
        )

        if params.persistence > 0.0:
            detail_noise = rng.normal(0.0, 1.0, size=terrain.shape).astype(np.float32)
            detail_sigma = max(1.0, 10.0 / max(1, params.octaves))
            detail_noise = ndimage.gaussian_filter(detail_noise, sigma=detail_sigma)
            noise_span = float(detail_noise.max() - detail_noise.min())
            if noise_span > 0.0:
                detail_noise = (detail_noise - detail_noise.min()) / noise_span
                detail_noise = detail_noise * 2.0 - 1.0
                detail_strength = params.persistence * max(params.hill_height, params.mountain_height) * 0.18
                terrain += detail_noise * detail_strength

        terrain = np.maximum(terrain, 0.0)
        return terrain

    def generate(self, params: GeneratorParams):
        terrain = self.generate_feature_terrain(params)

        # base_height ist ein Offset ab NORMALNULL (0 m) nach oben.
        height_data = params.base_height + terrain
        height_data = np.clip(height_data, self.MIN_HEIGHT, self.MAX_HEIGHT)

        amplitude = max(params.hill_height, params.mountain_height)
        height_data = self.apply_smoothing(height_data, params)
        height_data = self._enforce_natural_base(height_data, amplitude)
        height_data = self._apply_edge_fade(height_data, amplitude)
        gray_map = self.height_to_gray_array(height_data)

        if params.contrast != 1.0:
            gray_float = gray_map.astype(np.float32)
            sea_level_gray = 40.0
            gray_float = np.clip((gray_float - sea_level_gray) * params.contrast + sea_level_gray, 0, 255)
            gray_map = np.rint(gray_float).astype(np.uint8)

        analysis_layers = self.compute_analysis_layers(height_data)
        return height_data, gray_map, analysis_layers
