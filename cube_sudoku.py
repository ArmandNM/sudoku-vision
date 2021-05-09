import cv2
import numpy as np
import imutils

from pytesseract import Output
from PIL import Image

from sudoku import Sudoku

DEBUG_IMAGES = True


class CubeSudoku(Sudoku):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self, grid_mask):
        indices = np.where(grid_mask != 0)
        
        x1 = np.min(indices[1])
        y1 = np.min(indices[0])
        
        x2 = np.max(indices[1])
        y2 = np.max(indices[0])
        
        # Add some safety padding
        padding = 0
        return [x1 - padding, y1 - padding, x2 + padding, y2 + padding]

    def extract_aligned_grid(self, img_orig, img_lines, grid_mask):
        indices = np.where(grid_mask != 0)

        x_top = indices[1][np.argmin(indices[0])]        
        y_top = indices[0][np.argmin(indices[0])]
        
        x_left = indices[1][np.argmin(indices[1])]        
        y_left = indices[0][np.argmin(indices[1])]
        
        x_right = indices[1][np.argmax(indices[1])]        
        y_right = indices[0][np.argmax(indices[1])]
        
        x_bot = indices[1][np.argmax(indices[0])]        
        y_bot = indices[0][np.argmax(indices[0])]
        
        # cv2.circle(img, (int(x_top), int(y_top)), radius=7, color=(70, 230, 70), thickness=-7)
        # cv2.circle(img, (int(x_left), int(y_left)), radius=7, color=(70, 230, 70), thickness=-7)
        # cv2.circle(img, (int(x_right), int(y_right)), radius=7, color=(70, 230, 70), thickness=-7)
        # cv2.circle(img, (int(x_bot), int(y_bot)), radius=7, color=(70, 230, 70), thickness=-7)
        
        # Extract only the current grid from the image
        bbox = self.get_bounding_box(grid_mask)
        grid_img_blue = img_lines[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        grid_img_black = img_orig[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # cv2.imshow('corners', grid_img_blue)
        # cv2.waitKey(0)
        
        # Correct horizontal rotation
        theta1 = np.rad2deg(np.arctan2(y_top - y_left, x_top - x_left))
        theta2 = np.rad2deg(np.arctan2(y_right - y_top, x_right - x_top))
        
        if np.abs(theta1) < np.abs(theta2):
            horizontal_rotation = theta1
        else:
            horizontal_rotation = theta2
            
        grid_img_blue = imutils.rotate(grid_img_blue, angle=horizontal_rotation)
        grid_img_black = imutils.rotate(grid_img_black, angle=horizontal_rotation)
        
        # cv2.imshow('rotated', grid_img_blue)
        # cv2.waitKey(0)
        
        # Select only the grid region and remove empty space caused by rotation
        bbox = self.get_bounding_box(np.all(grid_img_blue == [255, 0, 0], axis=2))
        aligned_grid_img = grid_img_black[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # cv2.imshow('aligned', aligned_grid_img)
        # cv2.waitKey(0)
        
        return aligned_grid_img

    def solve(self, img, filename, output_path):
        img = self.resize_image(img, scale_percent=200)
        img_clean = img.copy()
        
        # Hough lines detection
        lines = self.detect_lines(img, filter_non_black=False)
        
        for line in lines:
            x1, y1, x2, y2, _, _, _ = line
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
        # if DEBUG_IMAGES:
        #     cv2.imshow('lines', img)
        #     cv2.waitKey(0)
        
        # Mask of grid lines
        grid_lines = 255 * np.all(img == [255, 0, 0], axis=2).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(grid_lines, connectivity=4)
        
        aligned_grid_img1 = self.extract_aligned_grid(img_clean, img, output == 1)
        aligned_grid_img2 = self.extract_aligned_grid(img_clean, img, output == 2)
        aligned_grid_img3 = self.extract_aligned_grid(img_clean, img, output == 3)
        
        if DEBUG_IMAGES:
            cv2.imshow('grid1', aligned_grid_img1)
            cv2.waitKey(0)
            cv2.imshow('grid2', aligned_grid_img2)
            cv2.waitKey(0)
            cv2.imshow('grid3', aligned_grid_img3)
            cv2.waitKey(0)


def main():
    sudoku = CubeSudoku()
    img = cv2.imread('datasets/train/cube/1.jpg')
    
    sudoku.solve(img, '2', './')


if __name__ == '__main__':
    main()
