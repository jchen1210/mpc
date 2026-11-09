import numpy as np
from dataclasses import dataclass

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class Fan:
    def __init__(self, bounds, direction, color, name):
        self.bounds    = np.array(bounds, dtype=float)
        d              = np.array(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        self.color     = color
        self.name      = name

    def contains(self, r):
        r = np.asarray(r)
        if r.ndim == 1:
            return all(self.bounds[i, 0] <= r[i] <= self.bounds[i, 1] for i in range(3))

        lower_ok = (r >= self.bounds[:, 0]) 
        upper_ok = (r <= self.bounds[:, 1])
        return np.all(lower_ok & upper_ok, axis=-1)


@dataclass
class Level:
    N:         int
    H:         int
    dt:        float
    ALPHA_MAX: float
    r0:        np.ndarray
    v0:        np.ndarray
    r_goal:    np.ndarray
    fans:      list


def load_level(path: str) -> Level:
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)

    p = cfg['params']
    s = cfg['start']

    fans = [
        Fan(fan['bounds'], fan['direction'], fan['color'], fan['name'])
        for fan in cfg.get('fans', [])
    ]

    return Level(
        N         = p['N'],
        H         = p['H'],
        dt        = p['dt'],
        ALPHA_MAX = p['alpha_max'],
        r0        = np.array(s['r0'],     dtype=float),
        v0        = np.array(s['v0'],     dtype=float),
        r_goal    = np.array(s['r_goal'], dtype=float),
        fans      = fans,
    )
