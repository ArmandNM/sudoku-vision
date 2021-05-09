import cv2
import numpy as np
import imutils

from os.path import join

from shapely.geometry import LineString, Point
from shapely import affinity


class Sudoku:
    def __init__(self):
        pass

    def resize_image(self, img, scale_percent=20):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        return img

    def detect_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 200)
        edges = cv2.Canny(gray, 50, 200, apertureSize=3)
        # cv2.imshow('edges_canny', edges)
        # cv2.waitKey(0)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=3)
        # cv2.imshow('edges_dilate', edges)
        # cv2.waitKey(0)

        # kernel = np.ones((3,3),np.uint8)
        edges = cv2.erode(edges, kernel, iterations=3)
        # cv2.imshow('edges_erode', edges)
        # cv2.waitKey(0)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                                minLineLength=300, maxLineGap=5)
        
        # Add Hough parameters to the detected lines
        hough_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = np.arctan2(y2 - y1, x1 - x2)
            theta_deg = np.rad2deg(theta)
            rho = x1 * np.sin(theta) + y1 * np.cos(theta)
            hough_lines.append([x1, y1, x2, y2, theta, theta_deg, rho])
        
        return hough_lines

    def merge_lines(self, lines):  # [x1, y1, x2, y2, theta, theta_deg, rho]
        # Sort lines by rho
        lines = sorted(lines, key=lambda line: line[-1])
        
        # print('\n\nSorted\n')
        # for line in lines:
        #     print(line)
        
        # Initialize all lines as ungrouped
        selected = np.zeros(len(lines), dtype=bool)

        grouped_lines = []
        
        # Group lines with similar rho and theta
        for it1, line1 in enumerate(lines):
            # Skip if line1 was already selected for merging
            if selected[it1]:
                continue

            # Create a new entry as a list of lines to be merged
            grouped_lines.append(np.empty((0, 7)))
            selected[it1] = True
            
            # Add current line as the first element in the new group
            grouped_lines[-1] = np.append(grouped_lines[-1], np.array([line1]), axis=0)
            
            for it2, line2 in enumerate(lines[it1+1:]):
                # Convert from relative to absolute index
                it2 = it1 + it2 + 1

                # Skip if line2 was already selected for merging
                if selected[it2]:
                    continue
                
                # Compare rho (must be closer than 5 pixels)
                if np.abs(line2[-1] - line1[-1]) > 5:
                    continue

                # Compare theta (must differ with at most 2 degrees)
                if np.abs(line2[-2] - line1[-2]) > 2:
                    continue

                # If all conditions are met, add line2 to the current group
                grouped_lines[-1] = np.append(grouped_lines[-1], np.array([line2]), axis=0)
                selected[it2] = True

        merged_lines = []

        for lines in grouped_lines:
            # Extract all points that represent ends of a segment
            points = np.vstack([lines[:, 0:2], lines[:, 2:4]])

            # Determine line orientation [0 - more horizontal, 1 - more vertical]
            rotation = lines[0, -2]  # theta_deg
            
            # Apply Q1 - Q3 and Q2 - Q4 equivallence
            if rotation > 90:
                rotation -= 180
            elif rotation < -90:
                rotation += 180
            
            if rotation <= 45 and rotation >= -45:  # Horizontal, pick leftmost and rightmost ends
                closest_axis = 0  # Ox
            elif rotation > 45 or rotation < -45:  # Vertical, pick uppermost and lowermost
                closest_axis = 1  # Oy

            # Find leftmost and rightmost points (or uppermost and lowermost)
            first_end_idx = points.argmin(axis=0)[closest_axis]
            second_end_idx = points.argmax(axis=0)[closest_axis]
            
            x1, y1 = points[first_end_idx]
            x2, y2 = points[second_end_idx]
            
            theta = np.arctan2(y2 - y1, x1 - x2)
            theta_deg = np.rad2deg(theta)
            rho = x1 * np.sin(theta) + y1 * np.cos(theta)
            merged_lines.append([x1, y1, x2, y2, theta, rotation, rho, closest_axis])

        return merged_lines

    def filter_lines(self, lines):
        # Remove lines that are not perpendicular on many other lines
        # Alternatively, remove lines outside the largest connected component
        filtered_lines = []
        intersection_points = []
        
        for it1, line1 in enumerate(lines):
            intersections = 0
            linestring1 = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
            
            for it2, line2 in enumerate(lines):
                if it1 == it2:
                    continue
                
                # Skip if lines don't intersect
                linestring2 = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
                if not linestring1.intersects(linestring2):
                    continue
                
                # Count intersections with perpendicular lines
                if np.abs(np.abs(line1[5] - line2[5]) % 180 - 90) < 3:  # theta_deg
                    intersections += 1
                
                # Save unique intersection points
                if it2 > it1:
                    intersection_points.append(linestring1.intersection(linestring2))
            
            # Keep line if it is perpendicular on many other lines
            if intersections > 3:
                filtered_lines.append(line1)

        return filtered_lines, intersection_points
    
    def determine_grid_rotation(self, lines):
        # Sort lines by theta
        lines = sorted(lines, key=lambda line: line[5])  # theta_deg
        
        grouped_lines = []
        # grouped_lines.append(np.empty((0, 8)))
        # grouped_lines[0] = np.append(grouped_lines[0], np.array([lines[0]]), axis=0)
        grouped_lines.append(np.array([lines[0]]))
        
        for line in lines[1:]:
            # Group lines with similar theta_deg
            if np.abs(line[5] - grouped_lines[-1][0, 5]) < 3:
                grouped_lines[-1] = np.append(grouped_lines[-1], np.array([line]), axis=0)
            else:
                grouped_lines.append(np.array([line]))
        
        assert len(grouped_lines) == 2
        # if len(grouped_lines) != 2:
        #     print('Something is not right!')

        # Take the rotation on the horizontal axis
        if grouped_lines[0][0, -1] == 0:
            grid_rotation = grouped_lines[0][:, 5].mean()
        elif grouped_lines[1][0, -1] == 0:
            grid_rotation = grouped_lines[1][:, 5].mean()
        
        return grid_rotation
    
    def determine_corners(self, intersection_points):
        points_matrix = np.array(list(map(lambda point: [point.x, point.y], intersection_points)))
        
        x_topleft, y_topleft = points_matrix.min(axis=0)
        x_botright, y_botright = points_matrix.max(axis=0)
        
        return x_topleft, y_topleft, x_botright, y_botright
    
    def is_cell_empty(self, img_patch):
        n_positive_pixels = (img_patch != 0).sum()
        n_negative_pixels = (img_patch == 0).sum()
        
        ink_ratio = n_positive_pixels / (n_positive_pixels + n_negative_pixels)
        return ink_ratio < 0.07
    
    def check_cells_content(self, img, x_topleft, y_topleft, x_botright, y_botright):
        grid_H = y_botright - y_topleft
        grid_W = x_botright - x_topleft
        
        cell_h = grid_H / 9
        cell_w = grid_W / 9
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([100, 100, 100])
        black_ink_mask = cv2.inRange(img, lower_black, upper_black)
        
        grid = np.zeros((9, 9))
        
        identified_cells = img.copy()
        detected_digits = img.copy()
        
        # Iterate the sudoku grid
        for i in range(9):
            for j in range(9):
                # Top left coordinates of the cell
                cell_x1 = x_topleft + j * cell_w
                cell_y1 = y_topleft + i * cell_h
                
                # Bottom right coordinates of the cell
                cell_x2 = cell_x1 + cell_w
                cell_y2 = cell_y1 + cell_h
                
                # Shrink the cell to center the content and ignore eventual borders
                # that have the same color intensity as the digits and can be detected
                # as false positives
                alpha = 0.25
                cell_x1 += alpha * cell_w 
                cell_y1 += alpha* cell_h
                cell_x2 -= alpha * cell_w
                cell_y2 -= alpha * cell_h
                
                # Draw selected patch
                patch_color_empty = np.zeros(img.shape, np.uint8)
                cv2.rectangle(img=patch_color_empty, pt1=(int(cell_x1), int(cell_y1)), pt2=(int(cell_x2), int(cell_y2)),
                              color=(200, 200, 200), thickness=cv2.FILLED)
                
                identified_cells = cv2.addWeighted(identified_cells, 1.0, patch_color_empty, 0.25, 0)
                
                # Check if cell contains a digit
                if self.is_cell_empty(black_ink_mask[int(cell_y1):int(cell_y2), int(cell_x1):int(cell_x2)]):
                    detected_digits = cv2.addWeighted(detected_digits, 1.0, patch_color_empty, 0.25, 0)
                else:
                    grid[i, j] = 1  # save detection
                    patch_color_non_empty = np.zeros(img.shape, np.uint8)
                    cv2.rectangle(img=patch_color_non_empty, pt1=(int(cell_x1), int(cell_y1)), pt2=(int(cell_x2), int(cell_y2)),
                              color=(0, 200, 10), thickness=cv2.FILLED)
                    detected_digits = cv2.addWeighted(detected_digits, 1.0, patch_color_non_empty, 0.50, 0)
        
        return grid, identified_cells, black_ink_mask, detected_digits

    def solve(self, img, filename, output_path):
        img = self.resize_image(img, scale_percent=20)
        
        # Hough lines detection
        lines = self.detect_lines(img)
        
        for line in lines:
            x1, y1, x2, y2, _, _, _ = line
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
        cv2.imwrite(join(output_path, '{}_sudoku_s1_hough.png'.format(filename)), img)
        
        # Combine lines with similar rho and theta
        merged_lines = self.merge_lines(lines)
        
        # Filter out lines that don't form a grid pattern (perpendicular on many other lines)
        filtered_lines, intersection_points = self.filter_lines(merged_lines)
        
        # Draw for easier visualization
        for line in filtered_lines:
            x1, y1, x2, y2, _, theta_deg, _, axis = line
            
            if np.abs(np.abs(theta_deg) % 90 - 45) < 5:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)  # ~45 degrees
            elif axis == 1:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)  # closer to vertical
            elif axis == 0:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)  # closer to horizontal

        # Mark all intersections between two lines
        for point in intersection_points:
            cv2.circle(img, (int(point.x), int(point.y)), radius=7, color=(70, 230, 70), thickness=-7)

        cv2.imwrite(join(output_path, '{}_sudoku_s2_filtered_and_dots.png'.format(filename)), img)
        
        try:
            horizontal_rotation = self.determine_grid_rotation(filtered_lines)
        except Exception:
            print(filename)
            return None, None
        
        # Rotate image around center to be aligned with Ox and Oy axes
        img = imutils.rotate(img, angle=-horizontal_rotation)
        
        (h, w) = img.shape[:2]
        (center_x, center_y) = (w / 2, h / 2)
        
        # Rotate intersection points around center to be aligned with Ox and Oy axes 
        intersection_points = list(map(lambda point: affinity.rotate(point, angle=horizontal_rotation, origin=(center_x, center_y)),
                                    intersection_points))
        
        # Draw the new position of the intersection points over the rotated image
        # If they are aligned with the previous dots drawn, both rotations are done correctly
        # (or wrong in the same way ^^)
        for point in intersection_points:
            cv2.circle(img, (int(point.x), int(point.y)), radius=5, color=(200, 150, 70), thickness=-5)
        
        cv2.imwrite(join(output_path, '{}_sudoku_s3_rotated.png'.format(filename)), img)

        # Draw estimated corners
        x_topleft, y_topleft, x_botright, y_botright = self.determine_corners(intersection_points)
        cv2.circle(img, (int(x_topleft), int(y_topleft)), radius=5, color=(30, 30, 255), thickness=-5)
        cv2.circle(img, (int(x_topleft), int(y_botright)), radius=5, color=(30, 30, 255), thickness=-5)
        cv2.circle(img, (int(x_botright), int(y_botright)), radius=5, color=(30, 30, 255), thickness=-5)
        cv2.circle(img, (int(x_botright), int(y_topleft)), radius=5, color=(30, 30, 255), thickness=-5)

        cv2.imwrite(join(output_path, '{}_sudoku_s4_corners.png'.format(filename)), img)
        
        grid, identified_cells, black_ink_mask, detected_digits = self.check_cells_content(img, x_topleft, y_topleft, x_botright, y_botright)
        cv2.imwrite(join(output_path, '{}_sudoku_s5_cells.png'.format(filename)), identified_cells)
        cv2.imwrite(join(output_path, '{}_sudoku_s6_mask.png'.format(filename)), black_ink_mask)
        cv2.imwrite(join(output_path, '{}_sudoku_s7_digits.png'.format(filename)), detected_digits)
                
        return grid, detected_digits


def main():
    sudoku = Sudoku()
    img = cv2.imread('datasets/train/classic/35.jpg')
    # img = cv2.imread('assets/custom/test_lines.jpg')
    
    sudoku.solve(img, '35', './')


if __name__ == '__main__':
    main()