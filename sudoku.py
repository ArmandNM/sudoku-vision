import cv2
import numpy as np


def merge_lines(lines):
    merged_lines = []
    pass


def detect_lines(img):
    scale_percent = 300 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
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
                            minLineLength=50, maxLineGap=5)
    
    # Add Hough parameters to the detected lines
    hough_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = np.arctan2(y2 - y1, x1 - x2)
        theta_deg = np.rad2deg(theta)
        rho = x1 * np.sin(theta) + y1 * np.cos(theta)
        hough_lines.append([x1, y1, x2, y2, theta, theta_deg, rho])
    
    for line in hough_lines:
        x1, y1, x2, y2, _, _, _ = line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    cv2.imwrite('sudoku.png', img)
    # cv2.imshow('sudoku', img)
    # cv2.waitKey(0)
    

def main():
    img = cv2.imread('assets/train/cube/5.jpg')
    detect_lines(img)


if __name__ == '__main__':
    main()