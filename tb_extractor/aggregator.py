from tb_extractor import extractor
from pathlib import Path
import numpy as np
import pandas as pd

def upsample(*frames):
    indices = np.arange(1 + max([frame.index.max() for frame in frames]))
    return [frame.astype(float).reindex(indices).interpolate('index', limit_direction='both') if frame.index.size > 1 else frame for frame in frames ]

def downsample(*frames, samples):
    max_ = 1 + max([frame.index.max() for frame in frames])
    indices = np.arange(max_, step=max_/samples)[1:]
    indices = np.append(indices, max_ - 1)
    return [frame.astype(float).reindex(indices) for frame in frames]

def aggregator(*frames):
    frames = upsample(*frames)
    if len(frames) > 1:
        return pd.concat(frames).groupby(level=0).agg(['count', np.mean, np.std])
    else: 
        return frames[0]
