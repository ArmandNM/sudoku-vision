import cv2
import numpy as np
import imutils

from os import listdir, makedirs
from os.path import isfile, exists, join, splitext

from shapely import affinity

from sudoku import Sudoku


def solve_task1(img, filename, output_path):
    sudoku = Sudoku()
    
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    
    lines = sudoku.detect_lines(img)
    
    for line in lines:
        x1, y1, x2, y2, _, _, _ = line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
    cv2.imwrite(join(output_path, '{}_sudoku_s1_hough.png'.format(filename)), img)
    
    merged_lines = sudoku.merge_lines(lines)
    filtered_lines, intersection_points = sudoku.filter_lines(merged_lines)
    
    for line in filtered_lines:
        x1, y1, x2, y2, _, theta_deg, _, axis = line
        
        if np.abs(np.abs(theta_deg) % 90 - 45) < 5:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)
        elif axis == 1:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
        elif axis == 0:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    for point in intersection_points:
        cv2.circle(img, (int(point.x), int(point.y)), radius=7, color=(70, 230, 70), thickness=-7)

    cv2.imwrite(join(output_path, '{}_sudoku_s2_filtered_and_dots.png'.format(filename)), img)
    
    try:
        horizontal_rotation = sudoku.determine_grid_rotation(filtered_lines)
    except Exception:
        print(filename)
        return
    
    img = imutils.rotate(img, angle=-horizontal_rotation)
    
    (h, w) = img.shape[:2]
    (center_x, center_y) = (w / 2, h / 2)
    
    intersection_points = list(map(lambda point: affinity.rotate(point, angle=horizontal_rotation, origin=(center_x, center_y)),
                                   intersection_points))
        
    for point in intersection_points:
        cv2.circle(img, (int(point.x), int(point.y)), radius=5, color=(200, 150, 70), thickness=-5)
    
    cv2.imwrite(join(output_path, '{}_sudoku_s3_rotated.png'.format(filename)), img)

    x_topleft, y_topleft, x_botright, y_botright = sudoku.determine_corners(intersection_points)
    cv2.circle(img, (int(x_topleft), int(y_topleft)), radius=5, color=(30, 30, 255), thickness=-5)
    cv2.circle(img, (int(x_topleft), int(y_botright)), radius=5, color=(30, 30, 255), thickness=-5)
    cv2.circle(img, (int(x_botright), int(y_botright)), radius=5, color=(30, 30, 255), thickness=-5)
    cv2.circle(img, (int(x_botright), int(y_topleft)), radius=5, color=(30, 30, 255), thickness=-5)

    cv2.imwrite(join(output_path, '{}_sudoku_s4_corners.png'.format(filename)), img)
    
    img = sudoku.check_cells_content(img, x_topleft, y_topleft, x_botright, y_botright)
    cv2.imwrite(join(output_path, '{}_sudoku_s5_cells.png'.format(filename)), img)


def main():
    input_path = 'assets/train/classic'
    output_path = 'results/train/classic'
    
    if not exists(output_path):
        makedirs(output_path)
    
    for file in listdir(input_path):
        img_path = join(input_path, file)
        if not isfile(img_path):
            continue
        
        # Skip non-image files
        filename, extension = splitext(file)
        if extension != '.jpg':
            continue
        
        # Load image
        img = cv2.imread(img_path)
        
        # Run task
        solve_task1(img, filename, output_path)


if __name__ == '__main__':
    main()
