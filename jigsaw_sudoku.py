import cv2
import numpy as np
import imutils

from shapely import affinity

from sudoku import Sudoku


class JigsawSudoku(Sudoku):
    def __init__(self):
        super().__init__()
    
    def extract_borders(self, img, x_topleft, y_topleft, x_botright, y_botright):
        grid_H = y_botright - y_topleft
        grid_W = x_botright - x_topleft
        
        cell_h = grid_H / 9
        cell_w = grid_W / 9
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([100, 100, 100])
        borders_mask = cv2.inRange(img, lower_black, upper_black)
        
        # cv2.imshow('borders', borders_mask)
        # cv2.waitKey(0)
        
        # Determine jigsaw type (colored or uncolored)
        # Colored has thicker borders and needs more erosion
        center_rect = img[int(x_topleft+grid_W/2-100):int(x_topleft+grid_W/2+100),
                          int(y_topleft+grid_H/2-100):int(y_topleft+grid_H/2+100)] 
        
        gray_percent = (cv2.cvtColor(center_rect, cv2.COLOR_BGR2HSV)[:, :, 1] < 10).sum() / (center_rect.shape[0] * center_rect.shape[1])
        
        if gray_percent > 0.7:
            iterations = 12
        else:
            iterations = 16
        
        # cv2.imshow('center', center_rect)
        # cv2.waitKey(0)
        
        # Erode thin lines
        bigger_borders_mask = self.resize_image(borders_mask, scale_percent=800)
        kernel = np.ones((3,3),np.uint8)
        bigger_borders_mask = cv2.erode(bigger_borders_mask, kernel, iterations=iterations)
        borders_mask = cv2.resize(bigger_borders_mask, (borders_mask.shape[1], borders_mask.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Pick largest connected components 
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(borders_mask, connectivity=4)
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
        
        borders_mask = 255 * (output == max_label).astype(np.uint8)
        
        # cv2.imshow('borders_largest', borders_mask)
        # cv2.waitKey(0)
        
        return borders_mask
    
    def separate_regions(self, borders_mask, x_topleft, y_topleft, x_botright, y_botright):
        pass
    
    def solve(self, img):
        img = self.resize_image(img, scale_percent=20)
        clean_img = img.copy()
        
        lines = self.detect_lines(img)
        merged_lines = self.merge_lines(lines)
        filtered_lines, intersection_points = self.filter_lines(merged_lines)
        
        try:
            horizontal_rotation = self.determine_grid_rotation(filtered_lines)
        except Exception:
            # print(filename)
            return None
        
        
        img = imutils.rotate(img, angle=-horizontal_rotation)
        clean_img = imutils.rotate(clean_img, angle=-horizontal_rotation)
        
        (h, w) = img.shape[:2]
        (center_x, center_y) = (w / 2, h / 2)
        
        intersection_points = list(map(lambda point: affinity.rotate(point, angle=horizontal_rotation, origin=(center_x, center_y)),
                                    intersection_points))

        x_topleft, y_topleft, x_botright, y_botright = self.determine_corners(intersection_points)
        
        borders_mask = self.extract_borders(clean_img, x_topleft, y_topleft, x_botright, y_botright)
        
        return borders_mask
        

def main():
    sudoku = JigsawSudoku()
    img = cv2.imread('assets/train/jigsaw/19.jpg')
    sudoku.solve(img)


if __name__ == '__main__':
    main()
