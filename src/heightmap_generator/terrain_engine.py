import noise
import numpy as np
import heapq
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

    def compute_extended_layers(self, height_data: np.ndarray, params: GeneratorParams):
        analysis = self.compute_analysis_layers(height_data)
        buildable, settlements, paths, buildings = self._compute_human_layers(height_data, analysis, params)
        analysis["buildable"] = buildable
        analysis["settlements"] = settlements
        analysis["paths"] = paths
        analysis["buildings"] = buildings
        return analysis

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

    def _apply_shore_effects(
        self,
        height_data: np.ndarray,
        amplitude: float,
        seed: int,
        shore_type: str = "standard",
        shore_width: float = 0.5,
    ) -> np.ndarray:
        """Apply configurable shoreline effects on the actual land-water boundary.

        Unlike a simple map-edge fade, this works on the contour where terrain
        meets sea level. That makes differences between shore modes clearly
        visible even when terrain features are not touching the map border.
        """
        SLOPE_RATIO = 2.0
        rows, cols = height_data.shape

        # Base border width derived from amplitude and minimum map fraction
        min_border = max(4, int(min(rows, cols) * 0.08))
        amp_border = int(amplitude * SLOPE_RATIO)
        base_border = max(min_border, amp_border)
        base_border = min(base_border, max(1, min(rows, cols) // 2))

        # shore_width 0.5 → 1.0× (unchanged); 0.0 → 0.4×; 1.0 → 1.6×
        shore_scale = 0.4 + float(np.clip(shore_width, 0.0, 1.0)) * 1.2
        border = int(max(4, base_border * shore_scale))
        border = min(border, max(1, min(rows, cols) // 2))

        sea_level = 0.0
        land_mask = height_data > sea_level

        # If there is no shoreline in the tile, apply mode-specific edge fallback
        # so coastline presets remain visibly different.
        if not np.any(land_mask) or np.all(land_mask):
            y = np.arange(rows, dtype=np.float32)
            x = np.arange(cols, dtype=np.float32)
            y_dist = np.minimum(y, (rows - 1.0) - y)[:, None]
            x_dist = np.minimum(x, (cols - 1.0) - x)[None, :]
            edge_dist = np.minimum(y_dist, x_dist)

            rng_edge = np.random.default_rng(int(seed) + 9917)

            def _edge_noise(sigma: float, amp: float) -> np.ndarray:
                raw = rng_edge.standard_normal((rows, cols)).astype(np.float32)
                s = ndimage.gaussian_filter(raw, sigma=max(1.0, sigma))
                s_std = float(np.std(s)) or 1.0
                return (s / s_std) * amp

            if shore_type == "strand":
                d = np.clip(edge_dist + _edge_noise(border * 0.40, border * 0.35), 0.0, None)
                t = np.clip(d / float(border), 0.0, 1.0)
                blend = t ** 0.55
            elif shore_type == "kliff":
                d = np.clip(edge_dist + _edge_noise(border * 0.30, border * 0.45), 0.0, None)
                t = np.clip(d / float(border), 0.0, 1.0)
                blend = t ** 4.5
            elif shore_type == "zerklueftet":
                d = np.clip(
                    edge_dist
                    + _edge_noise(border * 0.70, border * 0.70)
                    + _edge_noise(border * 0.16, border * 0.28),
                    0.0,
                    None,
                )
                t = np.clip(d / float(border), 0.0, 1.0)
                blend = t * t * (3.0 - 2.0 * t)
                blend = np.clip(blend + _edge_noise(border * 0.18, 0.08), 0.0, 1.0)
            elif shore_type == "delta":
                d = np.clip(
                    edge_dist
                    + _edge_noise(border * 0.95, border * 0.85)
                    + _edge_noise(border * 0.20, border * 0.20),
                    0.0,
                    None,
                )
                t = np.clip(d / float(border), 0.0, 1.0)
                blend = t * t * (3.0 - 2.0 * t)
                channel = np.clip(_edge_noise(border * 0.60, 0.22), -0.35, 0.35)
                blend = np.clip(blend - np.maximum(0.0, channel), 0.0, 1.0)
            else:
                t = np.clip(edge_dist / float(border), 0.0, 1.0)
                blend = t * t * (3.0 - 2.0 * t)

            return np.clip(height_data * blend, self.MIN_HEIGHT, self.MAX_HEIGHT)

        # Distance to shoreline inside land and inside water.
        dist_land = np.asarray(ndimage.distance_transform_edt(land_mask), dtype=np.float32)
        dist_water = np.asarray(ndimage.distance_transform_edt(~land_mask), dtype=np.float32)
        signed_dist = dist_land - dist_water

        rng = np.random.default_rng(int(seed) + 7391)

        def _smooth_noise(sigma: float, amp: float) -> np.ndarray:
            raw = rng.standard_normal((rows, cols)).astype(np.float32)
            s = ndimage.gaussian_filter(raw, sigma=max(1.0, sigma))
            s_std = float(np.std(s)) or 1.0
            return (s / s_std) * amp

        # Use warped signed distance to perturb coastline shape.
        if shore_type == "strand":
            warp = _smooth_noise(border * 0.40, border * 0.35)
        elif shore_type == "kliff":
            warp = _smooth_noise(border * 0.30, border * 0.45)
        elif shore_type == "zerklueftet":
            warp = _smooth_noise(border * 0.70, border * 0.70) + _smooth_noise(border * 0.16, border * 0.28)
        elif shore_type == "delta":
            warp = _smooth_noise(border * 0.95, border * 0.85) + _smooth_noise(border * 0.20, border * 0.20)
        else:
            warp = np.zeros_like(signed_dist, dtype=np.float32)

        d = signed_dist + warp
        d_land = np.clip(d, 0.0, None)
        t = np.clip(d_land / float(border), 0.0, 1.0)

        if shore_type == "strand":
            coast_factor = t ** 0.55
        elif shore_type == "kliff":
            coast_factor = t ** 4.5
        elif shore_type == "zerklueftet":
            coast_factor = t * t * (3.0 - 2.0 * t)
            coast_factor = np.clip(coast_factor + _smooth_noise(border * 0.18, 0.08), 0.0, 1.0)
        elif shore_type == "delta":
            coast_factor = t * t * (3.0 - 2.0 * t)
            channel = np.clip(_smooth_noise(border * 0.60, 0.22), -0.35, 0.35)
            coast_factor = np.clip(coast_factor - np.maximum(0.0, channel), 0.0, 1.0)
        else:
            coast_factor = t * t * (3.0 - 2.0 * t)

        out = height_data.copy()
        # Only shape land near coastline; keep water as-is (0 m or below).
        near_coast_land = (land_mask) & (d_land <= float(border))
        out[near_coast_land] = height_data[near_coast_land] * coast_factor[near_coast_land]
        return np.clip(out, self.MIN_HEIGHT, self.MAX_HEIGHT)

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

    def _adjust_landmass_size(self, height_data: np.ndarray, landmass_scale: float) -> np.ndarray:
        """Scale the generated terrain footprint uniformly in X/Y around center.

        UI semantics: 50 = neutral size, 100 = 2x footprint, 25 = 0.5x footprint.
        This preserves overall shape character and changes primarily size.
        """
        setting = float(landmass_scale)
        if setting <= 2.0:
            # Legacy compatibility: old factor mode.
            scale = float(np.clip(setting, 0.25, 2.0))
        else:
            scale = float(np.clip(setting / 50.0, 0.20, 2.00))

        if abs(scale - 1.0) < 1e-6:
            return height_data

        rows, cols = height_data.shape
        cy = (rows - 1) * 0.5
        cx = (cols - 1) * 0.5

        y = np.arange(rows, dtype=np.float32)
        x = np.arange(cols, dtype=np.float32)
        Y, X = np.meshgrid(y, x, indexing="ij")

        # Inverse mapping: sample source at scaled-back coordinates.
        src_y = (Y - cy) / scale + cy
        src_x = (X - cx) / scale + cx

        out = ndimage.map_coordinates(
            height_data.astype(np.float32),
            [src_y, src_x],
            order=1,
            mode="constant",
            cval=0.0,
        )

        return np.clip(out, self.MIN_HEIGHT, self.MAX_HEIGHT)

    def _apply_coastal_erosion(self, height_data: np.ndarray, strength: float, amplitude: float, seed: int) -> np.ndarray:
        if strength <= 0.0:
            return height_data

        land_mask = height_data > 0.0
        if not np.any(land_mask) or np.all(land_mask):
            return height_data

        rows, cols = height_data.shape
        border = max(6, int(min(rows, cols) * (0.06 + 0.12 * strength)))
        dist_land = np.asarray(ndimage.distance_transform_edt(land_mask), dtype=np.float32)
        near_coast = land_mask & (dist_land <= float(border))
        if not np.any(near_coast):
            return height_data

        rng = np.random.default_rng(int(seed) + 17011)
        raw = rng.standard_normal((rows, cols)).astype(np.float32)
        noise = ndimage.gaussian_filter(raw, sigma=max(1.5, border * 0.45))
        noise_std = float(np.std(noise)) or 1.0
        noise = noise / noise_std

        coastal_fade = np.clip(1.0 - dist_land / float(border), 0.0, 1.0) ** 1.35
        erosion_mask = np.clip(noise, 0.0, None) * coastal_fade
        erosion_depth = erosion_mask * amplitude * (0.10 + 0.22 * strength)

        out = height_data.copy()
        out[near_coast] = np.maximum(0.0, out[near_coast] - erosion_depth[near_coast])

        smoothed = ndimage.gaussian_filter(out, sigma=0.9 + strength * 0.8)
        blend = coastal_fade * (0.22 + 0.38 * strength)
        out = out * (1.0 - blend) + smoothed * blend
        return np.clip(out, self.MIN_HEIGHT, self.MAX_HEIGHT)

    def _apply_ridges(self, height_data: np.ndarray, strength: float, amplitude: float) -> np.ndarray:
        if strength <= 0.0 or amplitude <= 0.0:
            return height_data

        land = np.clip(height_data, 0.0, None).astype(np.float32)
        if float(np.max(land)) <= 0.0:
            return height_data

        broad = ndimage.gaussian_filter(land, sigma=2.0)
        detail = np.clip(land - broad, 0.0, None)
        scale = float(np.percentile(detail, 98.0))
        if scale <= 1e-6:
            return height_data

        ridges = np.clip(detail / scale, 0.0, 1.0) ** 0.8
        ridge_add = ridges * amplitude * (0.05 + 0.22 * strength)

        out = height_data.copy()
        out += ridge_add
        return np.clip(out, self.MIN_HEIGHT, self.MAX_HEIGHT)

    def _apply_drainage_light(self, height_data: np.ndarray, strength: float, amplitude: float, seed: int) -> np.ndarray:
        if strength <= 0.0 or amplitude <= 0.0:
            return height_data

        land = np.clip(height_data, 0.0, None).astype(np.float32)
        if float(np.max(land)) <= 0.0:
            return height_data

        rng = np.random.default_rng(int(seed) + 23057)
        raw = rng.standard_normal(land.shape).astype(np.float32)
        flow_noise = ndimage.gaussian_filter(raw, sigma=max(2.0, min(land.shape) * 0.025))
        noise_std = float(np.std(flow_noise)) or 1.0
        flow_noise = flow_noise / noise_std

        # Zero crossings of a smooth field create continuous tendril-like channel lines.
        channel_lines = np.exp(-((flow_noise / 0.22) ** 2)).astype(np.float32)
        grad_y, grad_x = np.gradient(land)
        slope_mag = np.sqrt(grad_x**2 + grad_y**2)
        slope_ref = float(np.percentile(slope_mag, 95.0))
        slope_norm = np.clip(slope_mag / max(1e-6, slope_ref), 0.0, 1.0)
        inland = np.clip(land / max(1e-6, float(np.percentile(land[land > 0.0], 95.0))), 0.0, 1.0)

        drainage_mask = channel_lines * (0.30 + 0.70 * slope_norm) * inland
        carve = drainage_mask * amplitude * (0.03 + 0.14 * strength)

        out = height_data.copy()
        land_mask = land > 0.0
        out[land_mask] = np.maximum(0.0, out[land_mask] - carve[land_mask])
        return np.clip(out, self.MIN_HEIGHT, self.MAX_HEIGHT)

    def _astar_path(
        self,
        cost_map: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        passable_mask: np.ndarray | None = None,
    ):
        def _astar_core(local_cost: np.ndarray, local_start: tuple[int, int], local_goal: tuple[int, int], local_mask: np.ndarray | None):
            rows, cols = local_cost.shape
            neighbors = [
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
                (-1, -1, 1.4142),
                (-1, 1, 1.4142),
                (1, -1, 1.4142),
                (1, 1, 1.4142),
            ]

            if local_mask is not None:
                if not bool(local_mask[local_start[0], local_start[1]]) or not bool(local_mask[local_goal[0], local_goal[1]]):
                    return []

            open_heap = []
            heapq.heappush(open_heap, (0.0, local_start))
            came_from = {}
            g_score = {local_start: 0.0}

            while open_heap:
                _, current = heapq.heappop(open_heap)
                if current == local_goal:
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path.reverse()
                    return path

                cy, cx = current
                for dy, dx, step_cost in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or nx < 0 or ny >= rows or nx >= cols:
                        continue
                    if local_mask is not None and not bool(local_mask[ny, nx]):
                        continue
                    tentative = g_score[current] + float(local_cost[ny, nx]) * step_cost
                    neighbor = (ny, nx)
                    if tentative >= g_score.get(neighbor, float("inf")):
                        continue
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    heuristic = abs(local_goal[0] - ny) + abs(local_goal[1] - nx)
                    heapq.heappush(open_heap, (tentative + heuristic, neighbor))

            return []

        rows, cols = cost_map.shape
        if max(rows, cols) >= 192:
            step = 2
            cost_small = cost_map[::step, ::step]
            mask_small = passable_mask[::step, ::step] if passable_mask is not None else None

            small_rows, small_cols = cost_small.shape
            s = (min(small_rows - 1, max(0, start[0] // step)), min(small_cols - 1, max(0, start[1] // step)))
            g = (min(small_rows - 1, max(0, goal[0] // step)), min(small_cols - 1, max(0, goal[1] // step)))

            def _nearest_passable(pt: tuple[int, int], mask: np.ndarray | None) -> tuple[int, int]:
                if mask is None or bool(mask[pt[0], pt[1]]):
                    return pt
                points = np.argwhere(mask)
                if points.size == 0:
                    return pt
                d = np.abs(points[:, 0] - pt[0]) + np.abs(points[:, 1] - pt[1])
                n = points[int(np.argmin(d))]
                return (int(n[0]), int(n[1]))

            s = _nearest_passable(s, mask_small)
            g = _nearest_passable(g, mask_small)
            coarse_path = _astar_core(cost_small, s, g, mask_small)
            if not coarse_path:
                return []

            full_path = []
            prev = None
            for cy, cx in coarse_path:
                fy = min(rows - 1, cy * step)
                fx = min(cols - 1, cx * step)
                if prev is None:
                    if passable_mask is None or bool(passable_mask[fy, fx]):
                        full_path.append((fy, fx))
                    prev = (fy, fx)
                    continue

                dy = fy - prev[0]
                dx = fx - prev[1]
                steps = max(abs(dy), abs(dx), 1)
                for i in range(1, steps + 1):
                    py = int(round(prev[0] + dy * (i / steps)))
                    px = int(round(prev[1] + dx * (i / steps)))
                    if passable_mask is not None and not bool(passable_mask[py, px]):
                        continue
                    if not full_path or full_path[-1] != (py, px):
                        full_path.append((py, px))
                prev = (fy, fx)

            return full_path

        return _astar_core(cost_map, start, goal, passable_mask)

    def _compute_human_layers(self, height_data: np.ndarray, analysis: dict, params: GeneratorParams):
        slope = analysis["slope"]
        roughness = analysis["roughness"]
        land = height_data > 0.0
        if not np.any(land):
            zeros = np.zeros_like(height_data, dtype=np.float32)
            return zeros, zeros, zeros, zeros

        coast_dist = np.asarray(ndimage.distance_transform_edt(land), dtype=np.float32)

        slope_limit = max(0.5, float(getattr(params, "buildable_max_slope", 8.0)))
        rough_limit = max(0.02, float(getattr(params, "buildable_max_roughness", 0.35)))
        coast_min = max(0.0, float(getattr(params, "min_coast_distance", 8.0)))

        slope_score = np.clip(1.0 - slope / slope_limit, 0.0, 1.0)
        rough_score = np.clip(1.0 - roughness / rough_limit, 0.0, 1.0)
        coast_score = np.clip((coast_dist - coast_min) / max(1.0, coast_min + 4.0), 0.0, 1.0)
        buildable = (0.45 * slope_score + 0.35 * rough_score + 0.20 * coast_score).astype(np.float32)
        buildable *= land.astype(np.float32)
        buildable[(slope > slope_limit) | (roughness > rough_limit) | (coast_dist < coast_min)] = 0.0
        if not getattr(params, "buildable_enabled", True):
            buildable = np.zeros_like(height_data, dtype=np.float32)

        if not getattr(params, "settlements_enabled", True):
            zeros = np.zeros_like(height_data, dtype=np.float32)
            return buildable, zeros, zeros.copy(), zeros.copy()

        settlement_size = float(np.clip(getattr(params, "settlement_size", 0.45), 0.0, 1.0))
        center_count = max(1, int(getattr(params, "settlement_count", 3)))

        # Fixed footprint requirements from UI request (meters ~= pixels).
        settlement_side = 50
        settlement_half = settlement_side // 2
        building_side = 15
        building_half = building_side // 2
        border_margin_input = max(0.0, float(getattr(params, "settlement_border_margin", 8.0)))

        candidate_map = ndimage.gaussian_filter(buildable, sigma=2.0 + settlement_size * 5.0)
        settlement_mask = np.zeros_like(height_data, dtype=np.float32)
        centers = []
        separation = max(settlement_side, int(min(height_data.shape) * (0.08 + settlement_size * 0.07)))
        work = candidate_map.copy()
        work_rows = int(work.shape[0])
        work_cols = int(work.shape[-1])

        # Keep full settlement footprints away from outer map border.
        border_margin = max(
            settlement_half + int(round(border_margin_input)),
            int(min(height_data.shape) * 0.02),
        )
        if work_rows > border_margin * 2 and work_cols > border_margin * 2:
            work[:border_margin, :] = 0.0
            work[-border_margin:, :] = 0.0
            work[:, :border_margin] = 0.0
            work[:, -border_margin:] = 0.0

        for _ in range(center_count):
            center_index = int(np.argmax(work))
            cy, cx = np.unravel_index(center_index, work.shape)
            if float(work[cy, cx]) < 0.12:
                break
            centers.append((cy, cx))
            radius = max(8, int(separation))
            y0 = max(0, cy - radius)
            y1 = min(work_rows, cy + radius + 1)
            x0 = max(0, cx - radius)
            x1 = min(work_cols, cx + radius + 1)
            work[y0:y1, x0:x1] = 0.0

        # Settlement area as fixed 50x50 square footprint.
        for cy, cx in centers:
            y0 = max(0, cy - settlement_half)
            y1 = min(work_rows, cy + settlement_half)
            x0 = max(0, cx - settlement_half)
            x1 = min(work_cols, cx + settlement_half)
            block = (buildable[y0:y1, x0:x1] > 0.08).astype(np.float32)
            settlement_mask[y0:y1, x0:x1] = np.maximum(settlement_mask[y0:y1, x0:x1], block)
        settlement_mask = np.clip(settlement_mask, 0.0, 1.0)

        if not getattr(params, "paths_enabled", True):
            zeros = np.zeros_like(height_data, dtype=np.float32)
            return buildable, settlement_mask, zeros, zeros.copy()

        path_mask = np.zeros_like(height_data, dtype=np.float32)
        slope_ref = float(np.percentile(slope, 95.0))
        rough_ref = float(np.percentile(roughness, 95.0))
        slope_norm = np.clip(slope / max(1e-6, slope_ref), 0.0, 1.0)
        rough_norm = np.clip(roughness / max(1e-6, rough_ref), 0.0, 1.0)
        water_penalty = (~land).astype(np.float32) * 40.0
        path_cost = 1.0 + 3.5 * slope_norm + 2.0 * rough_norm + 1.8 * (1.0 - buildable) + water_penalty
        path_curviness = float(np.clip(getattr(params, "path_curviness", 0.7), 0.0, 1.0))
        chaikin_iterations = int(round(path_curviness * 3.0))
        sample_density = 0.8 + path_curviness * 2.2

        land_points = np.argwhere(land)

        def _nearest_land_point(target):
            if land_points.size == 0:
                return target
            d = np.abs(land_points[:, 0] - target[0]) + np.abs(land_points[:, 1] - target[1])
            nearest = land_points[int(np.argmin(d))]
            return (int(nearest[0]), int(nearest[1]))

        def _nearest_path_point(target):
            points = np.argwhere(path_mask > 0.0)
            if points.size == 0:
                return target
            d = np.abs(points[:, 0] - target[0]) + np.abs(points[:, 1] - target[1])
            nearest = points[int(np.argmin(d))]
            return (int(nearest[0]), int(nearest[1]))

        def _chaikin(points, iterations=2):
            pts = [np.array([float(p[0]), float(p[1])], dtype=np.float32) for p in points]
            if len(pts) < 3:
                return points
            if iterations <= 0:
                return points
            for _ in range(max(1, iterations)):
                out = [pts[0]]
                for i in range(len(pts) - 1):
                    p = pts[i]
                    q = pts[i + 1]
                    out.append((0.75 * p + 0.25 * q).astype(np.float32))
                    out.append((0.25 * p + 0.75 * q).astype(np.float32))
                out.append(pts[-1])
                pts = out
            smooth = []
            for p in pts:
                y = int(np.clip(round(float(p[0])), 0, work_rows - 1))
                x = int(np.clip(round(float(p[1])), 0, work_cols - 1))
                smooth.append((y, x))
            return smooth

        def _draw_polyline(points):
            if len(points) < 2:
                return
            smoothed = _chaikin(points, iterations=chaikin_iterations)
            prev = smoothed[0]
            for cur in smoothed[1:]:
                dy = cur[0] - prev[0]
                dx = cur[1] - prev[1]
                steps = max(abs(dy), abs(dx), 1)
                for i in range(steps + 1):
                    py = int(round(prev[0] + dy * (i / steps)))
                    px = int(round(prev[1] + dx * (i / steps)))
                    if 0 <= py < work_rows and 0 <= px < work_cols and land[py, px]:
                        path_mask[py, px] = 1.0
                prev = cur

        def _sample_route(route, base_div):
            if not route:
                return []
            div = max(4, int(round(base_div * sample_density)))
            step = max(1, len(route) // div)
            sampled = route[::step]
            if sampled[-1] != route[-1]:
                sampled.append(route[-1])
            return sampled

        # Rounded ring roads around each settlement area (no right-angle boxes).
        ring_offset = settlement_half + 5
        for cy, cx in centers:
            n = max(24, int(2.0 * np.pi * ring_offset * (0.45 + 0.55 * path_curviness)))
            loop = []
            for i in range(n):
                a = 2.0 * np.pi * (i / n)
                # Slight ellipse variation avoids perfect geometric uniformity.
                amp = 0.02 + 0.10 * path_curviness
                ry = ring_offset * (0.95 + amp * np.sin(a * 2.0))
                rx = ring_offset * (1.00 + (amp * 0.8) * np.cos(a * 3.0))
                py = int(np.clip(round(cy + np.sin(a) * ry), 0, work_rows - 1))
                px = int(np.clip(round(cx + np.cos(a) * rx), 0, work_cols - 1))
                if land[py, px]:
                    loop.append((py, px))
            if len(loop) >= 3:
                loop.append(loop[0])
                _draw_polyline(loop)

        # Curved region loop through settlements sorted by polar angle.
        if len(centers) >= 3:
            my = float(np.mean([c[0] for c in centers]))
            mx = float(np.mean([c[1] for c in centers]))
            ordered = sorted(centers, key=lambda c: np.arctan2(c[0] - my, c[1] - mx))
            for i in range(len(ordered)):
                a = ordered[i]
                b = ordered[(i + 1) % len(ordered)]
                start = _nearest_land_point(a)
                goal = _nearest_land_point(b)
                route = self._astar_path(path_cost, start, goal, passable_mask=land)
                if route:
                    sampled = _sample_route(route, base_div=36)
                    _draw_polyline(sampled)

        # Connect settlements with main links and add a branch into each center.
        if len(centers) >= 2:
            connected = [centers[0]]
            for center in centers[1:]:
                anchor = min(connected, key=lambda pt: abs(pt[0] - center[0]) + abs(pt[1] - center[1]))
                start = _nearest_land_point(anchor)
                goal = _nearest_land_point(center)
                route = self._astar_path(path_cost, start, goal, passable_mask=land)
                if route:
                    sampled = _sample_route(route, base_div=40)
                    _draw_polyline(sampled)
                connected.append(center)

        for center in centers:
            anchor = _nearest_path_point(center)
            start = _nearest_land_point(anchor)
            goal = _nearest_land_point(center)
            route = self._astar_path(path_cost, start, goal, passable_mask=land)
            if route:
                sampled = _sample_route(route, base_div=24)
                _draw_polyline(sampled)

        # --- Einzelne Gebaeude ausserhalb von Ortschaften ---
        building_count = max(0, int(getattr(params, "building_count", 5)))
        building_mask = np.zeros_like(height_data, dtype=np.float32)
        building_centers = []
        if building_count > 0 and np.any(buildable > 0.05):
            settlement_exclusion = np.zeros_like(height_data, dtype=np.float32)
            settlement_radius_excl = settlement_half + 8
            for cy, cx in centers:
                y0 = max(0, cy - settlement_radius_excl)
                y1 = min(work_rows, cy + settlement_radius_excl + 1)
                x0 = max(0, cx - settlement_radius_excl)
                x1 = min(work_cols, cx + settlement_radius_excl + 1)
                settlement_exclusion[y0:y1, x0:x1] = 1.0
            build_candidate = buildable.copy()
            build_candidate[settlement_exclusion > 0.5] = 0.0

            # Keep building footprints away from map border as well.
            b_margin = max(
                building_half + int(round(border_margin_input * 0.8)),
                int(min(height_data.shape) * 0.015),
            )
            if work_rows > b_margin * 2 and work_cols > b_margin * 2:
                build_candidate[:b_margin, :] = 0.0
                build_candidate[-b_margin:, :] = 0.0
                build_candidate[:, :b_margin] = 0.0
                build_candidate[:, -b_margin:] = 0.0

            build_work = ndimage.gaussian_filter(build_candidate, sigma=1.5)
            min_sep = max(12, int(min(height_data.shape) * 0.05))
            placed = 0
            while placed < building_count:
                idx = int(np.argmax(build_work))
                by, bx = np.unravel_index(idx, build_work.shape)
                if float(build_work[by, bx]) < 0.05:
                    break
                y0h = max(0, by - building_half)
                y1h = min(work_rows, by + building_half + 1)
                x0h = max(0, bx - building_half)
                x1h = min(work_cols, bx + building_half + 1)
                block = (buildable[y0h:y1h, x0h:x1h] > 0.05).astype(np.float32)
                if float(np.sum(block)) > 0.0:
                    building_mask[y0h:y1h, x0h:x1h] = np.maximum(building_mask[y0h:y1h, x0h:x1h], block)
                    building_centers.append((int(by), int(bx)))
                y0 = max(0, by - min_sep); y1 = min(work_rows, by + min_sep + 1)
                x0 = max(0, bx - min_sep); x1 = min(work_cols, bx + min_sep + 1)
                build_work[y0:y1, x0:x1] = 0.0
                placed += 1

        # Branches from road network to every individual building center.
        for b_center in building_centers:
            anchor = _nearest_path_point(b_center)
            start = _nearest_land_point(anchor)
            goal = _nearest_land_point(b_center)
            route = self._astar_path(path_cost, start, goal, passable_mask=land)
            if route:
                sampled = _sample_route(route, base_div=18)
                _draw_polyline(sampled)

        # Wege strikt auf Land halten.
        path_mask *= land.astype(np.float32)

        path_width = max(1, int(round(float(getattr(params, "path_width", 2.0)))))
        if path_width > 1:
            path_mask = ndimage.binary_dilation(path_mask > 0.0, iterations=path_width - 1).astype(np.float32)
            path_mask *= land.astype(np.float32)

        # Final softening to suppress residual hard corners.
        path_soft = ndimage.gaussian_filter(path_mask.astype(np.float32), sigma=0.45 + 0.9 * path_curviness)
        path_mask = (path_soft > (0.30 - 0.12 * path_curviness)).astype(np.float32)
        path_mask *= land.astype(np.float32)

        return buildable.astype(np.float32), settlement_mask.astype(np.float32), path_mask.astype(np.float32), building_mask.astype(np.float32)

    def _apply_settlement_terraform(self, height_data: np.ndarray, analysis: dict, params: GeneratorParams | None = None) -> np.ndarray:
        buildable = np.asarray(analysis.get("buildable", 0.0), dtype=np.float32)
        settlements = np.asarray(analysis.get("settlements", 0.0), dtype=np.float32)
        buildings = np.asarray(analysis.get("buildings", 0.0), dtype=np.float32)
        paths = np.asarray(analysis.get("paths", 0.0), dtype=np.float32)

        if (
            buildable.shape != height_data.shape
            or settlements.shape != height_data.shape
            or buildings.shape != height_data.shape
            or paths.shape != height_data.shape
        ):
            return height_data

        out = height_data.astype(np.float32, copy=True)
        flat_ref = ndimage.gaussian_filter(out, sigma=4.0)
        buildable_strength = float(np.clip(getattr(params, "buildable_terraform_strength", 0.28), 0.0, 1.0))
        path_strength = float(np.clip(getattr(params, "path_terraform_strength", 1.0), 0.0, 1.0))

        # Weiche Kanten verhindern harte Stufen zwischen Terraforming und Bestand.
        buildable_w = np.clip(ndimage.gaussian_filter(buildable, sigma=2.3), 0.0, 1.0)
        settlement_w = np.clip(ndimage.gaussian_filter(settlements, sigma=2.0), 0.0, 1.0)
        building_w = np.clip(ndimage.gaussian_filter(buildings, sigma=1.3), 0.0, 1.0)
        path_w = np.clip(ndimage.gaussian_filter(paths, sigma=1.1), 0.0, 1.0)

        # Bauflächen planieren, bevor harte Siedlungs-Footprints einwirken.
        out = out * (1.0 - buildable_w * buildable_strength) + flat_ref * (buildable_w * buildable_strength)

        # Ortschaften stark planieren (50x50-Flaechen), Haeuser noch staerker.
        out = out * (1.0 - settlement_w * 0.90) + flat_ref * (settlement_w * 0.90)
        out = out * (1.0 - building_w * 0.97) + flat_ref * (building_w * 0.97)

        # Wege leicht eingraben und glätten, damit Trassen im Terrain lesbar bleiben.
        path_ref = ndimage.gaussian_filter(out, sigma=2.0)
        path_blend = 0.75 * path_strength
        path_carve = 1.20 * path_strength
        out = out * (1.0 - path_w * path_blend) + path_ref * (path_w * path_blend)
        out = out - path_w * path_carve

        # Keine kuenstlichen Landbruecken unter NN erzeugen.
        out = np.where(height_data > 0.0, np.maximum(0.0, out), out)
        return np.clip(out, self.MIN_HEIGHT, self.MAX_HEIGHT)

    def _create_shape_profile(self, shape_name: str, extent: float, rng: np.random.Generator) -> np.ndarray:
        min_extent = max(12.0, float(extent))
        base_radius = max(6.0, min_extent / 2.0)
        # Keep square axis-aligned; random rotation turns it into a diamond (Karo).
        if shape_name == "square":
            rotation = 0.0
        else:
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
            nx = xr / (radius_x * 1.35)
            ny = yr / (radius_y * 1.35)
            # Heart SDF: f <= 0 = Inneres der Herzform
            f = (nx**2 + ny**2 - 1.0) ** 3 - nx**2 * ny**3
            # Weiches Innen-Gewicht: steigt ab der Grenze schnell an
            inside_w = np.clip(-f * 5.0, 0.0, 1.0) ** 0.45
            # Kugelförmiges Höhenprofil (kein flaches Plateau) vom
            # Massenschwerpunkt des Herzens aus (liegt leicht unterhalb Mitte)
            dome_r = np.sqrt((nx * 0.82) ** 2 + ((ny - 0.12) * 0.82) ** 2)
            dome = np.clip(1.05 - dome_r, 0.0, 1.0) ** 0.52
            profile = (inside_w * dome).astype(np.float32)
            # Harte kreisförmige Maske – eliminiert Artefakte der quadratischen
            # Stempel-Bounding-Box und verhindert falsche "Land"-Pixel bei den
            # Küsteneffekten
            outer_r = np.sqrt((xr / (radius_x * 1.75)) ** 2 + (yr / (radius_y * 1.75)) ** 2)
            profile = np.clip(profile * np.clip(1.5 - outer_r, 0.0, 1.0), 0.0, 1.0).astype(np.float32)
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
        height_data = self._apply_ridges(height_data, getattr(params, "ridge_strength", 0.0), amplitude)
        height_data = self.apply_smoothing(height_data, params)
        height_data = self._enforce_natural_base(height_data, amplitude)
        if getattr(params, "shore_enabled", True):
            height_data = self._apply_shore_effects(
                height_data,
                amplitude,
                params.seed,
                getattr(params, "shore_type", "standard"),
                getattr(params, "shore_width", 0.5),
            )
        height_data = self._apply_coastal_erosion(
            height_data,
            getattr(params, "coastal_erosion_strength", 0.0),
            amplitude,
            params.seed,
        )
        height_data = self._apply_drainage_light(
            height_data,
            getattr(params, "drainage_strength", 0.0),
            amplitude,
            params.seed,
        )
        height_data = self._adjust_landmass_size(height_data, getattr(params, "landmass_scale", 50.0))
        analysis_layers = self.compute_extended_layers(height_data, params)
        if bool(getattr(params, "settlement_terraform_enabled", False)):
            height_data = self._apply_settlement_terraform(height_data, analysis_layers, params)
            analysis_layers = self.compute_extended_layers(height_data, params)

        gray_map = self.height_to_gray_array(height_data)

        if params.contrast != 1.0:
            gray_float = gray_map.astype(np.float32)
            sea_level_gray = 40.0
            gray_float = np.clip((gray_float - sea_level_gray) * params.contrast + sea_level_gray, 0, 255)
            gray_map = np.rint(gray_float).astype(np.uint8)

        return height_data, gray_map, analysis_layers
