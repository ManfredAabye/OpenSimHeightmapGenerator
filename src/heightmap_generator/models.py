from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorParams:
    width: int
    height: int
    hill_extent: float
    mountain_extent: float
    octaves: int
    persistence: float
    base_height: float
    hill_count: int
    hill_height: float
    mountain_count: int
    mountain_height: float
    seed: int
    terrain_type: str
    contrast: float
    gauss_sigma: float
    median_size: int
    auto_smooth: bool
    extra_smooth: bool
    landmass_scale: float = 50.0
    shore_type: str = "standard"
    shore_width: float = 0.5
    coastal_erosion_strength: float = 0.0
    ridge_strength: float = 0.0
    drainage_strength: float = 0.0
    buildable_max_slope: float = 8.0
    buildable_max_roughness: float = 0.35
    min_coast_distance: float = 8.0
    settlement_border_margin: float = 8.0
    settlement_count: int = 3
    building_count: int = 5
    settlement_size: float = 0.45
    path_width: float = 2.0
    path_curviness: float = 0.7
    settlement_terraform_enabled: bool = False
    buildable_terraform_strength: float = 0.28
    path_terraform_strength: float = 1.0
    shore_enabled: bool = True
    buildable_enabled: bool = False
    settlements_enabled: bool = False
    paths_enabled: bool = False
