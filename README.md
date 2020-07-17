# tb-extractor
## Usage
```console
Extracts data from tensorboard event files to CSV.
Usage: tb-extractor [OPTIONS] LOGDIR OUTPUT_DIR

  Extracts scalar data from tensorboard event files to CSV.

  LOGDIR: Path to event file or directory of event file. Directory if in
  recursive mode. OUTPUT_DIR: Path of output file. Directory if in recursive
  mode.

Options:
  -r, --recursive
  --block TEXT     Tags to block.
  --name TEXT      Name of csv in recursive mode.
  --everything     Extract all data, otherwise downsampled according to
                   Tensorboard backend defaults.
  -h, --help       Show this message and exit.
```
