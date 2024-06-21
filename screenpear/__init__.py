import click
from pprint import pprint
import cv2
import numpy as np
import datetime
import os
from pathlib import Path
import easyocr
import time

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

    # preprocess(src, dst)
    ocr_data = ocr_image(src)


    # Write red boxes around detected text
    image = cv2.imread(src)

    for ocr_box in ocr_data:
        top_left = tuple([int(val) for val in ocr_box[0][0]])
        bottom_right = tuple([int(val) for val in ocr_box[0][2]])
        # Extract the bounding box region
        box_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Separate text and background
        text_mask, background_mask = separate_text_background(box_region)


        # Get the dominant color of the text region
        text_color = get_dominant_color(box_region, text_mask)
        print(f'{ocr_box[1]}: Background color: {text_color}')

        # Get the dominant color of the background region
        background_color = get_dominant_color(box_region, background_mask)
        print(f'{ocr_box[1]}: Background color: {background_color}')

        # Draw the rectangle with the text color
        cv2.rectangle(image, top_left, bottom_right, text_color.tolist(), 2)

        # Optionally, draw a smaller rectangle inside to show background color
        cv2.rectangle(image, (top_left[0] + 5, top_left[1] + 5), (bottom_right[0] - 5, bottom_right[1] - 5), background_color.tolist(), 2)

    image = draw_red_boxes(image, ocr_data)

    background_color = get_dominant_color(image)

    cv2.imwrite(dst, image)


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


def ocr_image(src):
    # Define your custom dictionary
    custom_dict = ['NutCorp', 'DevOps']
    allowlist = ['http://', 'https://', 'www.']


    custom_dict_variations = []
    for word in custom_dict:
        custom_dict_variations.append(word)
        custom_dict_variations.append(word.lower())
        custom_dict_variations.append(word.upper())
        custom_dict_variations.append(word.capitalize())

    # Initialize the reader
    reader = easyocr.Reader(['en'], gpu=True)

    # Define a function to match detected words with the custom dictionary
    def custom_decoder(recognized_text, custom_dict_variations):
        recognized_words = []
        for item in recognized_text:
            text = item[1]
            if text in custom_dict:
                recognized_words.append(item)
        return recognized_words

    # Start the timer
    start_time = time.time()
    # Read the image
    results = reader.readtext(src, detail=1, decoder='greedy', beamWidth=5)

    # for (bbox, text, prob) in results:
    #     print(f"Detected text: {text} (Confidence: {prob})")

    # End the timer
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time

    # Apply custom decoder
    # filtered_results = custom_decoder(results, custom_dict)

    print(f"Detected text in {duration:.2f} seconds")
    # Print the filtered results
    # for (bbox, text, prob) in filtered_results:
    #     print(f"Detected text: {text} (Confidence: {prob})")
    return results


def draw_red_boxes(image, results):
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    return image


def get_dominant_color(image, mask=None):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply the mask if provided
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    # Remove zero pixels if mask is applied
    if mask is not None:
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    # Convert to float32
    pixels = np.float32(pixels)
    # Define criteria, number of clusters (K) and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 1
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = palette[0].astype(int)
    return dominant_color

# Function to separate text and background
def separate_text_background(box_region):
    gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_mask = binary
    background_mask = cv2.bitwise_not(binary)
    return text_mask, background_mask


if __name__ == '__main__':
    cli()

