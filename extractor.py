from pathlib import Path

import click
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    DEFAULT_SIZE_GUIDANCE, STORE_EVERYTHING_SIZE_GUIDANCE, EventAccumulator)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def dataframe(path, block, everything=False):
    if everything:
        size_guidance = STORE_EVERYTHING_SIZE_GUIDANCE
    else:
        size_guidance = DEFAULT_SIZE_GUIDANCE

    event_acc = EventAccumulator(str(path), size_guidance=size_guidance)
    event_acc.Reload()

    scalar_decoder = lambda x: x.value
    scalar_frame = extract(
        event_acc,
        "scalars",
        EventAccumulator.Scalars,
        scalar_decoder,
        block_list=block,
    )

    tensor_decoder = lambda x: tf.make_ndarray(x.tensor_proto)
    tensor_frame = extract(
        event_acc,
        "tensors",
        EventAccumulator.Tensors,
        tensor_decoder,
        block_list=block,
    )

    return scalar_frame.join(tensor_frame, how="outer", sort=True)


def extract(
    event_accumulator, tag_category, event_list_fn, value_decoder, block_list=[],
):
    tags = [i for i in event_accumulator.Tags()[tag_category] if i not in block_list]

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


def path_callback(ctx, param, value):
    if value is not None:
        return Path(value)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("logdir", type=click.Path(readable=True))
@click.argument("output_dir", type=click.Path(writable=True), callback=path_callback)
@click.option("--block", type=str, multiple=True, help="Tags to block.")
@click.option("--allow", type=str, multiple=True, help="Tags to allow.")
@click.option(
    "--everything",
    type=bool,
    is_flag=True,
    default=False,
    help="Extract all data, otherwise downsampled.",
)
def main(logdir: str, output_dir: Path, **kwargs):
    """Extracts scalar data from tensorboard event files to CSV.

        LOGDIR: Path to event file or directory of event file
        
        OUTPUT_DIR: Path of output file.
    """
    frame = dataframe(logdir, **kwargs)

    output_dir.parent.mkdir(exist_ok=True, parents=True)
    frame.to_csv(output_dir)
