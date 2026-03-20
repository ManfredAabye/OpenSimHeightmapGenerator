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
    shore_type: str = "standard"
    shore_width: float = 0.5
