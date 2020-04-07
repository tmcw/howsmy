print("TURNED OFF")
# Stops accidental running of the file
exit()

from pathlib import Path
import os.path
import imutils
import glob
import cv2
import os


captcha_image_folder = "images"
output_folder = "extracted_chars_real"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(captcha_image_folder, "*"))

counts = {}

# loop over the image paths
for (i, image_file) in enumerate(captcha_image_files):
    print(f"[INFO] processing image {i + 1}/{len(captcha_image_files)}")

    # Since the filename contains the captcha text, grab the base filename as the text
    captcha_correct_text = Path(image_file).stem

    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[1] if imutils.is_cv3() else contours[0]

    char_image_regions = []

    # Now we can loop through each of the four contours and extract the char
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect chars that
        # are conjoined into one chunk
        if w / h > 1:
            # This contour is too wide to be a single char!
            # Split it in half into two char regions!
            half_width = int(w / 2)
            char_image_regions.append((x, y, half_width, h))
            char_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal char by itself
            char_image_regions.append((x, y, w, h))


    # Sort the detected char images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right char
    char_image_regions = sorted(char_image_regions, key=lambda x: x[0])
    # Save out each char as a single image
    for char_bounding_box, char_text in zip(char_image_regions, captcha_correct_text):
        # Grab the coordinates of the char in the image
        x, y, w, h = char_bounding_box
        # Extract the char from the original image with a 2-pixel margin around the edge
        char_image = gray[y:y + h, x:x + w]

        # Get the folder to save the image in
        save_path = os.path.join(output_folder, char_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the char image to a file
        count = counts.get(char_text, 1)

        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        print(f"char image: {char_image}")
        cv2.imwrite(p, char_image)

        # increment the count for the current key
        counts[char_text] = count + 1
