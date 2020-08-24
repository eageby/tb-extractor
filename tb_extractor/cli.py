from pathlib import Path
from typing import List
import click

import tb_extractor.extractor as extractor
import tb_extractor.aggregator as aggregator

def path_callback(ctx, param, value): 
    if value is not None:
        return Path(value)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("logdir", type=click.Path(readable=True))
@click.argument("output_dir", type=click.Path(writable=True), callback=path_callback)
@click.option("agg", "--aggregate", "-a", is_flag=True)
@click.option("--recursive", "-r", is_flag=True)
@click.option("--block", type=str, multiple=True, help="Tags to block.")
@click.option("samples", '--downsample', type=int, default=None)
@click.option("--name", type=str, default='extracted.csv', help="Name of csv in recursive/aggregate mode.")
@click.option(
    'everything',
    "--not-all/--all-samples",
    type=bool,
    is_flag=True,
    default=False,
    help="Downsampled according to Tensorboard backend defaults.",
)
@click.option(
    'metric',
    "--metric",
    type=bool,
    is_flag=True,
    help="To include indices < 0. Default behavior includes indices > 0.",
)
def main(logdir: str, output_dir: Path, recursive: bool, samples: int, agg: bool, name: str, block: List[str], metric: bool, **kwargs):
    """Extracts scalar data from tensorboard event files to CSV.

        LOGDIR: Path to event file or directory of event file. Directory if in recursive mode.
        OUTPUT_DIR: Path of output file. Directory if in recursive mode.
    """
    try: 
        if recursive or agg:
            logdir = Path(logdir)

            if logdir.is_file():
                if recursive:
                    mode_str = "Recursive "
                else:
                    mode_str = "Aggregate "
                raise ValueError(mode_str + "mode requires a directory.")

            event_dirs = list(set([i.parent for i in logdir.glob('**/*event*')]))
            frames = extractor.frames(*event_dirs, block=block, metric=metric) 
            if agg:
                aggregated = aggregator.aggregator(*frames)
                if isinstance(aggregated.columns[0], tuple):
                    aggregated.columns = aggregated.columns.map('_'.join)
                write = [(aggregated, output_dir/name)]
            else:
                output_dirs = [output_dir / i / name for i in event_dirs]
                write = zip(frames, output_dirs)
        else:
            write = [(extractor.dataframe(logdir,metric=metric, block=block), output_dir/name)]
            
        if write: 
            for frame, write_dir in write:
                if samples:
                    frame = aggregator.downsample(frame, samples=samples)[0]

                write_dir.parent.mkdir(exist_ok=True, parents=True)
                frame.to_csv(write_dir)

    except ValueError as err:
        print("Error:",  err)
