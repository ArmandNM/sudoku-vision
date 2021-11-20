import cv2
import numpy as np
import imutils
import torch

from os.path import join

from itertools import permutations

from sudoku import Sudoku
from digit_classifier import get_classifier

DEBUG_IMAGES = False


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

        # Extract only the current grid from the image
        bbox = self.get_bounding_box(grid_mask)
        grid_img_blue = img_lines[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        grid_img_black = img_orig[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Correct horizontal rotation
        theta1 = np.rad2deg(np.arctan2(y_top - y_left, x_top - x_left))
        theta2 = np.rad2deg(np.arctan2(y_right - y_top, x_right - x_top))

        if np.abs(theta1) < np.abs(theta2):
            horizontal_rotation = theta1
        else:
            horizontal_rotation = theta2

        grid_img_blue = imutils.rotate(grid_img_blue, angle=horizontal_rotation)
        grid_img_black = imutils.rotate(grid_img_black, angle=horizontal_rotation)

        # Select only the grid region and remove empty space caused by rotation
        bbox = self.get_bounding_box(np.all(grid_img_blue == [255, 0, 0], axis=2))
        padding = 5
        aligned_grid_blue = grid_img_blue[bbox[1]+padding:bbox[3]-padding, bbox[0]+padding:bbox[2]-padding]
        aligned_grid_black = grid_img_black[bbox[1]+padding:bbox[3]-padding, bbox[0]+padding:bbox[2]-padding]

        return aligned_grid_blue, aligned_grid_black

    def identify_digits(self, aligned_grid_img):
        digits = np.zeros((9, 9))
        model = get_classifier()

        grid_H, grid_W = aligned_grid_img.shape[:2]
        cell_h = grid_H / 9
        cell_w = grid_W / 9

        # Make digits white over black background as in MNIST
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([100, 100, 100])
        binary_grid = cv2.inRange(aligned_grid_img, lower_black, upper_black)
        kernel = np.ones((3, 3), np.uint8)
        binary_grid = cv2.dilate(binary_grid, kernel, iterations=1)

        for i in range(9):
            for j in range(9):
                # Top left coordinates of the cell
                cell_x1 = j * cell_w
                cell_y1 = i * cell_h

                # Bottom right coordinates of the cell
                cell_x2 = cell_x1 + cell_w
                cell_y2 = cell_y1 + cell_h

                # Take cell center coordinates
                alpha = 0.01
                cell_x1 += alpha * cell_w
                cell_y1 += alpha* cell_h
                cell_x2 -= alpha * cell_w
                cell_y2 -= alpha * cell_h

                # Extract patch
                patch = binary_grid[int(cell_y1):int(cell_y2), int(cell_x1):int(cell_x2)]
                patch = cv2.resize(patch, (28, 28), interpolation=cv2.INTER_CUBIC)

                # Convert to torch tensor
                patch = np.expand_dims(patch, axis=0)
                patch = np.expand_dims(patch, axis=0)
                patch = torch.Tensor(patch / 255)

                # Make prediction
                preds = torch.softmax(model(patch), dim=-1)
                digit = preds.argmax().item()
                digits[i, j] = digit

        return digits

    def is_grid_correct(self, digits):
        correct = True
        for i in range(1, 10):
            if (digits == i).sum() != 9:
                correct = False
        return correct

    def is_order_correct(self, digits_grid1, digits_grid2, digits_grid3):
        correct =  True

        if not np.all(digits_grid1[-1, :] == digits_grid2[0, :]):
            correct = False
        if not np.all(digits_grid2[:, -1] == digits_grid3[:, 0]):
            correct = False
        if not np.all(digits_grid1[:, -1] == np.flip(digits_grid3[0, :])):
            correct = False

        return correct

    def matching_error(self, digits_grid1, digits_grid2, digits_grid3):
        # Find number of unmatching digits on margins
        error = 0
        error += (digits_grid1[-1, :] != digits_grid2[0, :]).sum()
        error += (digits_grid2[:, -1] != digits_grid3[:, 0]).sum()
        error += (digits_grid1[:, -1] != np.flip(digits_grid3[0, :])).sum()

        return error

    def project_on_cube(self, face1, face2, face3):
        def project(face, region, template):
            white = 255 * np.ones((face.shape[0], face.shape[1], 3), np.uint8)
            face_points = np.float32([[0, 0], [face.shape[1], 0],
                                     [face.shape[1], face.shape[0]], [0, face.shape[0]]])
            h, _ = cv2.findHomography(face_points, region)

            white_warped = cv2.warpPerspective(white, h, (template.shape[1], template.shape[0]))
            face_warped = cv2.warpPerspective(face, h, (template.shape[1], template.shape[0]))

            template = template.astype(float) - white_warped.astype(float) + face_warped.astype(float)
            template = template.astype(np.uint8)

            if DEBUG_IMAGES:
                cv2.imshow('face warped', template)
                cv2.waitKey(0)

            return template

        template_W = 588
        template_H = 544
        template = 255 * np.ones((template_H, template_W, 3), dtype=np.uint8)

        A = [5, 158]
        B = [5, 463]
        C = [296, 539]
        D = [296, 234]
        E = [564, 386]
        F = [564, 83]
        G = [272, 2]

        region2 = np.float32([A, D, C, B])
        template = project(face2, region2, template)

        region3 = np.float32([D, F, E, C])
        template = project(face3, region3, template)

        region1 = np.float32([G, F, D, A])
        template = project(face1, region1, template)

        return template

    def solve(self, img, filename, output_path):
        img = self.resize_image(img, scale_percent=200)
        img_clean = img.copy()

        # Hough lines detection
        lines = self.detect_lines(img, filter_non_black=False)

        for line in lines:
            x1, y1, x2, y2, _, _, _ = line
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Mask of grid lines
        grid_lines = 255 * np.all(img == [255, 0, 0], axis=2).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(grid_lines, connectivity=4)

        aligned_grid_blue1, aligned_grid_black1 = self.extract_aligned_grid(img_clean, img, output == 1)
        aligned_grid_blue2, aligned_grid_black2 = self.extract_aligned_grid(img_clean, img, output == 2)
        aligned_grid_blue3, aligned_grid_black3 = self.extract_aligned_grid(img_clean, img, output == 3)

        cv2.imwrite(join(output_path, '{}_cube_grid1.png'.format(filename)), aligned_grid_black1)
        cv2.imwrite(join(output_path, '{}_cube_grid2.png'.format(filename)), aligned_grid_black2)
        cv2.imwrite(join(output_path, '{}_cube_grid3.png'.format(filename)), aligned_grid_black3)

        digits_grid1 = self.identify_digits(aligned_grid_blue1)
        digits_grid2 = self.identify_digits(aligned_grid_blue2)
        digits_grid3 = self.identify_digits(aligned_grid_blue3)
        if self.is_grid_correct(digits_grid1) and \
           self.is_grid_correct(digits_grid2) and \
           self.is_grid_correct(digits_grid3):
               print('Grid digits seem correct!')
        else:
            print('Some digits might be detected wrong, but I\'ll try to find the most likely face ordering! No worries!')

        # Backtracking to find the right face order
        digit_grids = [digits_grid1, digits_grid2, digits_grid3]
        faces = [aligned_grid_black1, aligned_grid_black2, aligned_grid_black3]
        perm = list(permutations([0, 1, 2]))
        sols = []

        for (idx1, idx2, idx3) in perm:
            error = self.matching_error(digit_grids[idx1], digit_grids[idx2], digit_grids[idx3])
            sols.append([idx1, idx2, idx3, error])

        # Find the face ordering with minimum error
        sols.sort(key=lambda sol: sol[-1])
        idx1, idx2, idx3, _ = sols[0]
        result = self.project_on_cube(faces[idx1], faces[idx2], faces[idx3])
        cv2.imwrite(join(output_path, '{}_result.png'.format(filename)), result)

        return digit_grids[idx1], digit_grids[idx2], digit_grids[idx3]


def main():
    sudoku = CubeSudoku()
    img = cv2.imread('datasets/train/cube/1.jpg')

    sudoku.solve(img, '1', './')


if __name__ == '__main__':
    main()
