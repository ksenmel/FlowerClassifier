import cv2
import numpy as np


def contrast_flower(image: np.ndarray) -> float:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray_image)


def amount_of_green(image):
    """
    Индикатор для одуванчиков (рядом много зеленой растительности)
    :param image:
    :return:
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    return cv2.countNonZero(green_mask)


def amount_of_yellow(image: np.ndarray) -> int:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    return cv2.countNonZero(yellow_mask)


def amount_of_red(image: np.ndarray) -> int:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    return cv2.countNonZero(red_mask)


def amount_of_white(image: np.ndarray) -> int:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    return cv2.countNonZero(white_mask)


def avg_saturation(image: np.ndarray) -> float:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Переводим изображение в HSV
    saturation = hsv_image[:, :, 1]  # Извлекаем канал насыщенности
    return float(np.mean(saturation))


def avg_brightness(image: np.ndarray) -> float:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Переводим изображение в HSV
    brightness = hsv_image[:, :, 2]  # Извлекаем канал яркости
    return float(np.mean(brightness))


feature_functions = {
    "amount_of_yellow": amount_of_yellow,
    "amount_of_red": amount_of_red,
    "amount_of_white": amount_of_white,
    "avg_saturation": avg_saturation,
    "avg_brightness": avg_brightness,

}
