from tb_extractor import extractor
from pathlib import Path
import numpy as np
import pandas as pd

def resample(*frames):
    indices = np.arange(1 + max([frame.index.max() for frame in frames]))
    return [frame.reindex(indices).interpolate('index', limit_direction='both') for frame in frames]

def aggregator(*frames):
    frames = resample(*frames)
    return pd.concat(frames).groupby(level=0).agg(['size', np.mean, np.std])
