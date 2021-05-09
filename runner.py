import cv2
import numpy as np
import imutils
import argparse
import difflib
import sys

from os import listdir, makedirs
from os.path import isfile, exists, join, splitext

from shapely import affinity

from sudoku import Sudoku
from jigsaw_sudoku import JigsawSudoku


RUN_CHECKER = True


def check(gt_path, prediction_path):
    with open(prediction_path, 'r') as f:
        pred = f.readlines()
    with open(gt_path, 'r') as f:
        gt = f.readlines()
        
    # Remove newlines
    pred = [line.strip() for line in pred]
    gt = [line.strip() for line in gt]
        
    diff = difflib.unified_diff(pred, gt, fromfile=prediction_path, tofile=gt_path)
    
    correct = True
    for line in diff:
        correct = False
    
    return correct


def solve_task1(img, filename, output_path):
    def _write_result(grid, filepath):
        with open(filepath, 'w') as f:
            for i in range(9):
                crt_line = ''
                for j in range(9):
                    if grid[i, j] != 0:
                        crt_line += 'x'
                    else:
                        crt_line += 'o'
                f.write(crt_line + '\n')
    
    sudoku = Sudoku()    
    grid, _ = sudoku.solve(img, filename, output_path)
    
    if grid is None:
        print(filename)
    else:
        _write_result(grid, join(output_path, '{}_predicted.txt'.format(filename)))


def solve_task2(img, filename, output_path):
    def _write_result(grid, regions, filepath):
        with open(filepath, 'w') as f:
            for i in range(9):
                crt_line = ''
                for j in range(9):
                    crt_line += str(int(regions[i, j]))
                    if grid[i, j] != 0:
                        crt_line += 'x'
                    else:
                        crt_line += 'o'
                f.write(crt_line + '\n')
    
    sudoku = JigsawSudoku()
    grid, regions = sudoku.solve(img, filename, output_path)
    
    if grid is None:
        print(filename)
    else:
        _write_result(grid, regions, join(output_path, '{}_predicted.txt'.format(filename)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='datasets/train/jigsaw')
    parser.add_argument('--output_path', default='results/train/jigsaw')
    parser.add_argument('--task', default=2)
    args = parser.parse_args()
    
    print(args)
    
    assert args.task in [1, 2, 3]
    
    tasks = [solve_task1, solve_task2]
    
    if not exists(args.output_path):
        makedirs(args.output_path)
    
    if RUN_CHECKER:
        total_tests = 0
        correct_tests = 0
    
    for file in listdir(args.input_path):
        img_path = join(args.input_path, file)
        if not isfile(img_path):
            continue
        
        # Skip non-image files
        filename, extension = splitext(file)
        if extension != '.jpg':
            continue
        
        # Load image
        img = cv2.imread(img_path)
        
        # Run task
        tasks[args.task - 1](img, filename, args.output_path)
        
        # Compare with gt
        if RUN_CHECKER:
            total_tests += 1
            try:
                if check(gt_path=join(args.input_path, '{}_gt.txt'.format(filename)),
                        prediction_path=join(args.output_path, '{}_predicted.txt'.format(filename))):
                    print(f'Test {filename} is correct!')
                    correct_tests += 1
                else:
                    print(f'\nErrors in test {filename}!\n')
            except:
                print(f'\nErrors in test {filename}!\n')


    if RUN_CHECKER:
        print(f'Summary: you got {correct_tests}/{total_tests} correct tests.')

if __name__ == '__main__':
    main()
