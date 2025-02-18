#  Copyright 2025 Dmitry Kireev <dmitry@atd.sh>
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

import Vision
import AppKit

def ocr_image(image_path):
    """Performs OCR on an image using Apple's Vision framework."""

    # Load image as NSImage
    ns_image = AppKit.NSImage.alloc().initWithContentsOfFile_(image_path)
    if ns_image is None:
        print("Failed to load image.")
        return

    # Convert NSImage to CGImage
    image_data = ns_image.TIFFRepresentation()
    bitmap = AppKit.NSBitmapImageRep.alloc().initWithData_(image_data)
    if bitmap is None:
        print("Failed to create bitmap representation.")
        return

    cg_image = bitmap.CGImage()
    if cg_image is None:
        print("Failed to convert image to CGImage.")
        return

    # Create a VNImageRequestHandler with the CGImage
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})

    # Create a text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)  # High accuracy

    # Perform the request
    success, error = handler.performRequests_error_([request], None)
    if not success:
        print(f"Failed to perform OCR: {error}")
        return

    # Extract recognized text
    extracted_text = []
    for observation in request.results():
        top_candidate = observation.topCandidates_(1).firstObject()
        if top_candidate:
            extracted_text.append(top_candidate.string())

    # Print extracted text
    if extracted_text:
        print("\n".join(extracted_text))
    else:
        print("No text detected.")

# Run OCR on the given image
if __name__ == "__main__":
    image_path = "/Users/dmitry/dev/automationd/lab.atd.sh/screenpear/data/input/digitalocean-01-input.jpg"
    ocr_image(image_path)
