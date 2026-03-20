import noise
import numpy as np
from matplotlib.path import Path
from scipy import ndimage

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

        if shape_name == "triangle":
            nx = xr / radius_x
            ny = yr / radius_y
            points = np.column_stack((nx.ravel(), ny.ravel()))
            triangle = Path([(0.0, -1.0), (-0.92, 0.70), (0.92, 0.70)])
            profile = triangle.contains_points(points).reshape(nx.shape).astype(np.float32)
        else:
            assert power is not None
            r = (np.abs(xr / radius_x) ** power + np.abs(yr / radius_y) ** power) ** (1.0 / power)
            profile = np.clip(1.0 - r, 0.0, 1.0).astype(np.float32)

        round_sigma = max(1.0, max(radius_x, radius_y) * 0.08)
        profile = ndimage.gaussian_filter(profile, sigma=round_sigma)
        max_profile = float(np.max(profile))
        if max_profile > 0.0:
            profile = profile / max_profile
        return np.clip(profile, 0.0, 1.0) ** 1.6

    def _place_features(self, terrain: np.ndarray, count: int, height_value: float, extent: float, shape_name: str, rng: np.random.Generator):
        rows, cols = terrain.shape
        effective_extent = max(float(extent), float(height_value) * 4.0)
        stamp = self._create_shape_profile(shape_name, effective_extent, rng) * float(height_value)
        stamp_h, stamp_w = stamp.shape
        half_h = stamp_h // 2
        half_w = stamp_w // 2
        margin_y = min(max(half_h + 2, int(height_value * 2.0)), max(1, rows // 2))
        margin_x = min(max(half_w + 2, int(height_value * 2.0)), max(1, cols // 2))

        for _ in range(max(0, int(count))):
            if cols > margin_x * 2:
                center_x = int(rng.integers(margin_x, cols - margin_x))
            else:
                center_x = cols // 2
            if rows > margin_y * 2:
                center_y = int(rng.integers(margin_y, rows - margin_y))
            else:
                center_y = rows // 2

            x0 = max(0, center_x - half_w)
            y0 = max(0, center_y - half_h)
            x1 = min(cols, center_x + half_w + 1)
            y1 = min(rows, center_y + half_h + 1)

            sx0 = x0 - (center_x - half_w)
            sy0 = y0 - (center_y - half_h)
            sx1 = sx0 + (x1 - x0)
            sy1 = sy0 + (y1 - y0)
            terrain[y0:y1, x0:x1] += stamp[sy0:sy1, sx0:sx1]

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

        return height_data, gray_map
