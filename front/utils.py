import cv2
import numpy as np
import math


def get_skew_angles(img, kernel_size=5, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.atan((y1 - y2) / (x2 - x1))
            angles.append(angle)

    return angles


def get_angle(img, units="degrees"):
    angles = get_skew_angles(img=img)
    filtered_angles = [angle for angle in angles if abs(angle) < 0.25]
    skew_angle = sum(filtered_angles) / len(filtered_angles)
    skew_angle_in_degrees = skew_angle * 180 / math.pi
    return skew_angle_in_degrees if units == "degrees" else skew_angle


def rotate_image(image, angle: float):
    new_image = image.copy()
    (h, w) = new_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(new_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return new_image