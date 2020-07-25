import functools
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import (
    DEFAULT_SIZE_GUIDANCE, STORE_EVERYTHING_SIZE_GUIDANCE, EventAccumulator)

SCALAR_RESERVOIR = {
    "tag_category": "scalars",
    "event_list_fn": EventAccumulator.Scalars,
    "value_decoder": lambda x: x.value,
}

TENSOR_RESERVOIR = {
    "tag_category": "tensors",
    "event_list_fn": EventAccumulator.Tensors,
    "value_decoder": lambda x: tf.make_ndarray(x),
}

def frames(*directories, **kwargs):
    return [dataframe(d, **kwargs) for d in *directories]
    
def dataframe(path, block=[], reservoirs=[SCALAR_RESERVOIR, TENSOR_RESERVOIR], everything=False):
    if everything:
        size_guidance = STORE_EVERYTHING_SIZE_GUIDANCE
    else:
        size_guidance = DEFAULT_SIZE_GUIDANCE

    event_acc = EventAccumulator(str(path), size_guidance=size_guidance)
    event_acc.Reload()

    frames = [extract(event_acc, **res, block_list=block) for res in reservoirs]
    return functools.reduce(lambda a, b: a.join(b, how="outer", sort=True), frames)

def extract(
    event_accumulator, tag_category, event_list_fn, value_decoder, block_list=[]
):
    tags = [i for i in event_accumulator.Tags()[tag_category]
            if i not in block_list]

    runlog_data = None

    for tag in tags:
        event_list = event_list_fn(event_accumulator, tag)

        values = list(map(value_decoder, event_list))
        step = list(map(lambda x: x.step, event_list))

        frame = {"step": step, tag: values}
        frame = pd.DataFrame(frame).set_index("step")

        if runlog_data is not None:
            runlog_data = runlog_data.join(frame)
        else:
            runlog_data = frame

    if runlog_data is None:
        return pd.DataFrame(None)

    return runlog_data
