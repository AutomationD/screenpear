import click
from pprint import pprint
import cv2
import numpy as np
import datetime
import os
from pathlib import Path


@click.group()
def cli():
    pass


@cli.command()
@click.option('--src', help='')
@click.option('--dst', help='')
def ocr(src, dst):
    date = datetime.datetime.now()
    data_path = os.path.join(os.getcwd(), 'data')
    print(f"data_path: {data_path}")

    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} not found')

    if dst is None:
        dst = os.path.join(data_path, 'output', f'{Path(src).stem}-{date:%Y-%m-%dT%H%M%S}{Path(src).suffix}')

    if Path(dst).is_dir():
        if not os.path.exists(dst):
            os.mkdir(dst)

        dst = os.path.join(dst, f'{Path(src).stem}-{date:%Y-%m-%dT%H%M%S}{Path(src).suffix}')

    print(f"src: {src}")
    print(f"dst: {dst}")

    preprocess(src, dst)


def preprocess(input_path, output_path):
    # Read the image
    image = cv2.imread(input_path)
    #######
    # Preprocess the image
    #######

    # Save the result
    cv2.imwrite(output_path, image)
    print(output_path)


if __name__ == '__main__':
    cli()

