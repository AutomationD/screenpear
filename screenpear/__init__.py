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

        dst = os.path.join(data_path, 'output', f'{Path(src).stem}-{date:%Y-%m-%dT%H%M%S}{Path(src).suffix}'.replace('input', 'output'))

    if Path(dst).is_dir():
        if not os.path.exists(dst):
            os.mkdir(dst)

        dst = os.path.join(dst, f'{Path(src).stem}-{date:%Y-%m-%dT%H%M%S}{Path(src).suffix}')

    print(f"src: {src}")
    print(f"dst: {dst}")

    preprocess(src, dst)


def preprocess(src, dst):
    # Read the image
    image = cv2.imread(src)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to create a black and white image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours to detect areas
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for areas with dark background
    mask = np.zeros_like(gray)

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Assume areas with dark background have text that is lighter, hence need to be inverted
        # Check the average color within this bounding box in the original image
        roi = gray[y:y + h, x:x + w]
        avg_color = np.mean(roi)

        # If the average color is dark, we consider this as an area with a dark background
        if avg_color < 128:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Invert areas in the mask
    inverted = cv2.bitwise_not(binary, mask=mask)

    # Combine the inverted areas with the original binary image
    final_image = cv2.bitwise_or(binary, inverted)

    # Invert the final image to have black text on white background
    final_image = cv2.bitwise_not(final_image)

    # Save the result
    cv2.imwrite(dst, final_image)


if __name__ == '__main__':
    cli()



