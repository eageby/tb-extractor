from pathlib import Path
import click

import tb_extractor.extractor as extractor
import tb_extractor.aggregator as aggregator

def path_callback(ctx, param, value): if value is not None:
        return Path(value)

def path_callback(ctx, param, value):
    if value is not None:
        return Path(value)


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("logdir", type=click.Path(readable=True))
@click.argument("output_dir", type=click.Path(writable=True), callback=path_callback)
@click.option("--recursive", "-r", is_flag=True)
@click.option("--block", type=str, multiple=True, help="Tags to block.")
@click.option("--name", type=str, default='extracted.csv', help="Name of csv in recursive mode.")
@click.option(
    'everything',
    "--downsampled/--all-samples",
    type=bool,
    is_flag=True,
    default=False,
    help="Downsampled according to Tensorboard backend defaults.",
)

def main(logdir: str, output_dir: Path, recursive: bool, name: str, **kwargs):
    """Extracts scalar data from tensorboard event files to CSV.

        LOGDIR: Path to event file or directory of event file. Directory if in recursive mode.
        OUTPUT_DIR: Path of output file. Directory if in recursive mode.
    """
    if recursive or aggregator:
        logdir = Path(logdir)
        event_dirs = list(set([i.parent for i in logdir.glob('**/*event*')]))
        frames = extractor.frames(*event_dirs) 

        if aggregator:
            aggregated = aggregator.aggregator(*frames)
            write = [(aggregated, output_dir)]
        else:
            output_dirs = [output_dir / i / name for i in event_dirs]
            write = zip(frames, output_dirs)
    else:
        write = (extractor.frames(*[logdir][0], output_dir)

    for frame, write_dir in write:
        write_dir.parent.mkdir(exist_ok=True, parents=True)
        frame.to_csv(write_dir)
