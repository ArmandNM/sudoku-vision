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
    
    def update_matrix(self, matrix):
        matrix[0, 0] = 1
    
    def separate_regions(self, clean_img, borders_mask, x_topleft, y_topleft, x_botright, y_botright):
        def _dfs(x, y, regions, n_regions):
            dx = [0, 1, 0, -1]
            dy = [1, 0, -1, 0]
            
            # Mark current cell as visited
            regions[y, x] = n_regions
            
            # Move recursively to neighbours
            for i in range(4):
                new_x = x + dx[i]
                new_y = y + dy[i]
                
                # Don't move outside the grid
                if new_x < 0 or new_x >= 9:
                    continue
                if new_y < 0 or new_y >= 9:
                    continue
                
                # Check if there is a border between
                center1_x = x_topleft + (x + 0.5) * cell_w
                center1_y = y_topleft + (y + 0.5) * cell_h
                
                center2_x = x_topleft + (new_x + 0.5) * cell_w
                center2_y = y_topleft + (new_y + 0.5) * cell_h
                
                if int(center1_x) > int(center2_x) or int(center1_y) > int(center2_y):
                    center1_x, center1_y, center2_x, center2_y = center2_x, center2_y, center1_x, center1_y
                
                if i in [0, 2]:  # vertical movement
                    passing_bar = borders_mask[int(center1_y):int(center2_y), int(center1_x-5):int(center2_x+5)]
                    clean_img[int(center1_y):int(center2_y), int(center1_x-5):int(center2_x+5)] = 255
                    print(passing_bar.shape)
                else:  # horizontal movement
                    passing_bar = borders_mask[int(center1_y-5):int(center2_y+5), int(center1_x):int(center2_x)]
                    clean_img[int(center1_y-5):int(center2_y+5), int(center1_x):int(center2_x)] = 255
                    print(passing_bar.shape)
                
                # cv2.imshow('midline', clean_img)
                # cv2.waitKey(0)
                border_ratio = (passing_bar != 0).sum() / (passing_bar.shape[0] * passing_bar.shape[1])
                
                # Don't move forward if cells are delimited by jigsaw border
                if border_ratio > 0.0:
                    print(border_ratio)
                    continue
                
                # Move to next unvisited cell
                if regions[new_y, new_x] == 0:
                    _dfs(new_x, new_y, regions, n_regions)
        
        grid_H = y_botright - y_topleft
        grid_W = x_botright - x_topleft
        
        cell_h = grid_H / 9
        cell_w = grid_W / 9
        
        n_regions = 0
        regions = np.zeros((9, 9))  
        
        for j in range(9):
            for i in range(9):
                if regions[j, i] == 0:
                    n_regions += 1
                    _dfs(i, j, regions, n_regions)
                    print(regions)
    
        print('done')
    
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
        
        self.separate_regions(clean_img, borders_mask, x_topleft, y_topleft, x_botright, y_botright)
        
        return borders_mask
        

def main():
    sudoku = JigsawSudoku()
    img = cv2.imread('assets/train/jigsaw/35.jpg')
    sudoku.solve(img)


if __name__ == '__main__':
    main()
