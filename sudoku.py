import cv2
import numpy as np


class Sudoku:
    def __init__(self):
        pass

    def merge_lines(self, lines):  # [x1, y1, x2, y2, theta, theta_deg, rho]
        # Sort lines by theta and rho
        lines = sorted(lines, key=lambda line: line[-1])
        
        print('\n\nSorted\n')
        for line in lines:
            print(line)
        
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
            merged_lines.append([x1, y1, x2, y2, theta, theta_deg, rho, closest_axis])

        return merged_lines

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
        
        for line in hough_lines:
            print(line)
            x1, y1, x2, y2, _, _, _ = line
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        cv2.imwrite('sudoku.png', img)
        cv2.imshow('sudoku', img)
        cv2.waitKey(0)

        return hough_lines


def main():
    sudoku = Sudoku()
    img = cv2.imread('assets/train/classic/5.jpg')
    # img = cv2.imread('assets/custom/test_lines.jpg')
    
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    
    lines = sudoku.detect_lines(img)
    merged_lines = sudoku.merge_lines(lines)
    
    print('\n\n\nconverted\n\n')
    for line in merged_lines:
        print(line)
        x1, y1, x2, y2, _, theta_deg, _, axis = line
        
        if np.abs(np.abs(theta_deg) % 90 - 45) < 5:
            print('oblic') 
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)
        elif axis == 1:
            print('vertical')
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
        elif axis == 0:
            print('orizontal')
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    cv2.imshow('sudoku_merged', img)
    cv2.imwrite('sudoku_merged.png', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()