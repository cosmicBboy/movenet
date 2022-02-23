"""Module for curating subset of the kinetics dataset."""

import shutil
from pathlib import Path

import typer
import yaml



app = typer.Typer()


def copy_file(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(src_path, dst_path)


@app.command()
def main(
    dataset_fp: Path,
    output_dataset_fp: Path,
    curation_metadata_fp: Path = typer.Option(...),
):
    with curation_metadata_fp.open() as fp:
        curation_metadata = yaml.safe_load(fp)

    typer.echo("Curating kinetics dataset")

    for split, categories in curation_metadata.items():
        for category, video_ids in categories.items():
            src_dir = dataset_fp / split / category
            dst_dir = output_dataset_fp / split / category
            for video_id in video_ids:
                copy_file(
                    (src_dir / video_id).with_suffix(".mp4"),
                    (dst_dir / video_id).with_suffix(".mp4"),
                )
        print(split, categories)

    typer.echo("Done")


if __name__ == "__main__":
    typer.run(main)
