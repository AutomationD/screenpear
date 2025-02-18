#  Copyright 2024 Dmitry Kireev <dmitry@atd.sh>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import click
from pprint import pprint
import cv2
import numpy as np
import datetime
import os
from pathlib import Path
import easyocr
import time
import re

# Define your custom dictionary
# TODO: word variations & replacements generation. For now just static
# TODO: Make this part of a config file


REPLACE_DICT = {
    'NutCorp': 'SquirrelCorp',
    'DevOps': 'SysAdmin',
    'nutcorp': 'squirrelcorp',
    'devops': 'sysadmin',
    # 'Search':'Find',
    'github': 'screenpear',
    'tesseract': 'EasyOCR',
    'postgres': 'sqlite',
    'Upgrade': 'update',
    'Orange': 'Raspberry',
    'metameetings': 'instabark',
    'mapdigital': 'barkdigital',
    'mm-core': 'ib-core',
    'mm-': 'ib-'
}

URL_PATTERNS = ['http://', 
                'https://',
                '.com', 
                'www.'
                ]


@click.group()
def cli():
    pass

@cli.command()
@click.option('--src', help='')
@click.option('--dst', help='')
@click.option('-w', '--width', help='')
@click.option('--url', is_flag=True, help='')



def ocr(src, dst, url, width=None):

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
    
    # Read image in nd array before passing it to easyocr
    image = cv2.imread(src)
    
    # Check the width argument (for desired image width) and perform resize accordingly
    if width is not None:
        image = resize(image, width)

    # preprocess(src, dst)
    ocr_data = ocr_image(image)

    # Extracting text detection result
    texts = []
    text_colors = []
    background_colors = []
    bboxes = []

    for idx, ocr_box in enumerate(ocr_data):

        # Text        
        text = ocr_box[1]
        texts.append(text)
        # Extract the bounding box region
        top_left = tuple([int(val) for val in ocr_box[0][0]])
        bottom_right = tuple([int(val) for val in ocr_box[0][2]])
        box_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        bboxes.append([top_left, bottom_right])

        # Get the dominant color of the text region
        text_color = np.array(get_text_color(box_region))
        text_colors.append(text_color)

        # Get the dominant color of the background region
        background_color = np.array(get_bg_color(box_region))
        background_colors.append(background_color)
    

    # Processing image based on text detection result
    for idx, txt in enumerate (texts):
        
        txt_new = txt
                
        # Word replacement on URL only
        if url:
            for url_pattern in URL_PATTERNS:
                if re.search (url_pattern.lower(), txt_new):
                    # Replacement lookup        
                    for replacement in REPLACE_DICT.items():
                        txt_new = txt_new.replace(replacement[0], replacement[1])
                    # Trim the URL result
                    txt_new = url_trim(txt_new)
                    break
        
        # Word replacement on all detected text
        else:
            # Replacement lookup        
            for replacement in REPLACE_DICT.items():
                txt_new = txt_new.replace(replacement[0], replacement[1])

        # Check if match and replacement is made. Otherwise, skip this text
        if txt_new == txt:
            continue

        print(f'{txt}: replaced_string={txt_new} text_color={text_colors[idx]}, background_color={background_colors[idx]}')

        # Draw the rectangle 
        # cv2.rectangle(image, bboxes[idx][0], bboxes[idx][1], [255,0,0], 2)

        # Get the text size before writing
        txt_size, txt_base = cv2.getTextSize(txt_new, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        txt_w, txt_h = txt_size

        # Get the font size. Ratio of text width and rectangle width
        box_w =  abs(bboxes[idx][0][0] - bboxes[idx][1][0])
        font_size = box_w / txt_w

        # Get the text size again after resizing before writing
        txt_size, txt_base = cv2.getTextSize(txt_new, cv2.FONT_HERSHEY_DUPLEX, font_size, 1)
        txt_w, txt_h = txt_size

        # Write the detected text above the bounding box
        ## Get height of the box
        box_h =  bboxes[idx][0][1] - bboxes[idx][1][1]
        ## Calculate text y offset
        text_offset_y = int(abs(box_h-txt_h) / 2)
        # Calculate the text corners
        txt_bot_left = (bboxes[idx][0][0], 
                        bboxes[idx][0][1]+text_offset_y)
        txt_top_right = (bboxes[idx][0][0] + txt_w, 
                         bboxes[idx][0][1]+text_offset_y-txt_h)
        
        cv2.rectangle(image, 
                      (txt_bot_left[0], txt_bot_left[1] + int(txt_h*0.3)), 
                      (txt_top_right[0], txt_top_right[1] - int(txt_h*0.3)), 
                      background_colors[idx].tolist(), 
                      cv2.FILLED
                      )
        
        cv2.putText(image, 
                    txt_new, 
                    txt_bot_left, 
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, 
                    text_colors[idx].tolist(), 
                    1,
                    cv2.LINE_AA)

    # Writing masked image to file        
    cv2.imwrite(dst, image)


def resize(img, target_w):
    # Resizing
    factor_x = int(target_w) / img.shape[1]
    img_resized = cv2.resize(img, (int(img.shape[1]* factor_x), int(img.shape[0]*factor_x)))
    return img_resized


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

    # Initialize the reader
    reader = easyocr.Reader(['en'], gpu=True)

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


def get_bg_color(image):
    # Get background color (the most dominant color)
    a2D = image.reshape(-1, image.shape[-1])
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def get_text_color(image):
    # Get text color (the 2nd dominant color)

    a2D = image.reshape(-1, image.shape[-1])
    a2D = np.float32(a2D)
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _, label, center=cv2.kmeans(a2D, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(res.T, col_range)
    txt_color = np.unravel_index(np.argsort(np.bincount(a1D))[-2], col_range)
    
    return txt_color


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


def url_trim(url_str):
    # Correcting possible errors
    word_to_trim = {' com': '.com', 'www ': 'www.' }
    for word_ in word_to_trim:
        txt_new = url_str.replace(word_, word_to_trim[word_])
    # white space stripping
    txt_new = txt_new.replace(' ', '')
    return txt_new
    

# Function to separate text and background
def separate_text_background(box_region):
    gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_mask = binary
    background_mask = cv2.bitwise_not(binary)
    return text_mask, background_mask


if __name__ == '__main__':
    cli()
