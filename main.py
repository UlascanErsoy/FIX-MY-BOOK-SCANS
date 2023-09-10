import cv2
import sys,os
import numpy as np
from typing import Tuple 

TARGET_WIDTH = 2480
TARGET_HEIGHT= 3508
Y_CROP_MARGIN = 50
X_CROP_MARGIN = 50
MIN_MARGIN_X = 100
MIN_MARGIN_Y = 150

def get_page_outer_bound(img) -> Tuple[int,int,int,int]:
    """
    Returns the outer bound
    x,y,w,h
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    outer_rect = [0,0,height,width]
    first = True 

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        #cv2.rectangle(img, (x, x+w), (y,y+h), (255,0,0), 2)
        if (x+w < 100):
            continue

        if w < 50 or h < 50:
            continue

        if first:
            outer_rect = [x,y,x+w,y+h]
            first = False
            continue

        outer_rect[0] = min(x, outer_rect[0])
        outer_rect[1] = min(y, outer_rect[1])
        outer_rect[2] = max(x+w, outer_rect[2])
        outer_rect[3] = max(y+h, outer_rect[3])

    #cv2.rectangle(img, (outer_rect[0], outer_rect[1]), (outer_rect[2],outer_rect[3]), (0,255,0), 2)

    return (outer_rect[0], outer_rect[1],
            outer_rect[2], outer_rect[3])

def centered_clean_page(img) -> np.ndarray:
    """
    return an image at the target size
    with the source image centered
    """
    new_img = np.full((TARGET_HEIGHT,TARGET_WIDTH,3), 255, dtype=np.uint8)

    (h, w) = img.shape[:2]

    margin_y = MIN_MARGIN_Y #(TARGET_HEIGHT - h) // 2
    margin_x = (TARGET_WIDTH  - w) // 2

    new_img[margin_y:img.shape[0]+margin_y,margin_x:margin_x+img.shape[1]] = img

    return new_img

if __name__ == "__main__":

    path = sys.argv[1]

    if not os.path.isdir(path):
        print("Please provide a path with the pictures!")

    files = os.listdir(path)
    
    output_path = os.path.join(path, "output")
    os.mkdir(output_path)
    
    for file in files:
        print(f"Processing {file}")
        try:
            img = cv2.imread(os.path.join(path, file))
        except Exception as e:
            print(f"Error reading image {e}! Skipping {file}")
            continue
   
        (height, width) = img.shape[:2]

        (x1,y1,x2,y2) = get_page_outer_bound(img)
        crop_img = img[y1-Y_CROP_MARGIN:y2+Y_CROP_MARGIN,x1-X_CROP_MARGIN:x2+X_CROP_MARGIN]

        new_img = centered_clean_page(crop_img)
        (height, width) = new_img.shape[:2]

        cv2.imwrite(os.path.join(output_path, file), new_img)

