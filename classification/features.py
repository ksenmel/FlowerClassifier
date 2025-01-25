import cv2
import numpy as np


def contrast_flower(image: np.ndarray) -> float:
    """
    Computes the standard deviation of brightness in grayscale image, which helps identify flowers like daisies or sunflowers.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: Standard deviation of brightness values.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray_image))  # type: ignore[arg-type]


def amount_of_green(image: np.ndarray) -> int:
    """
    Calculates the amount of green in the image, useful for identifying flowers surrounded by greenery (e.g., dandelions).

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        int: The number of pixels detected as green.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    return cv2.countNonZero(green_mask)


def amount_of_yellow(image: np.ndarray) -> int:
    """
    Calculates the amount of yellow in the image, helpful for identifying flowers like sunflowers.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        int: The number of pixels detected as yellow.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    return cv2.countNonZero(yellow_mask)


def amount_of_red(image: np.ndarray) -> int:
    """
    Calculates the amount of red in the image, useful for identifying flowers with red colors (e.g., roses).

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        int: The number of pixels detected as red.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    return cv2.countNonZero(red_mask)


def avg_saturation(image: np.ndarray) -> float:
    """
    Computes the average saturation of the image in HSV space, indicating how vivid the colors are.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: The average saturation value.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1]  # Extract the saturation channel
    return float(np.mean(saturation))


def avg_brightness(image: np.ndarray) -> float:
    """
    Computes the average brightness of the image in HSV space.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: The average brightness value.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv_image[:, :, 2]  # Extract the brightness channel
    return float(np.mean(brightness))


def circularity(image: np.ndarray) -> float:
    """
    Calculates the circularity of the object in the image, which can help identify round shapes like sunflowers.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: Circularity value (1 for perfect circle, closer to 0 for irregular shapes).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def bud_shape(image: np.ndarray) -> float:
    """
    Computes the elongation ratio of the object to identify flower bud shapes, typically elongated for certain flowers.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: The elongation ratio (higher values for elongated shapes).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0
    largest_contour = max(contours, key=cv2.contourArea)
    _, _, width, height = cv2.boundingRect(largest_contour)

    if height == 0:
        return 0.0
    return max(width, height) / min(width, height)


def count_objects(image: np.ndarray) -> int:
    """
    Counts the number of distinct objects in the image, which can help in identifying flowers like dandelions or tulips.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        int: The number of detected objects in the image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for contour in contours if cv2.contourArea(contour) > 500)


def contour_complexity(image: np.ndarray) -> float:
    """
    Computes the fractal complexity of contours in the image, useful for identifying complex petal shapes in flowers like roses.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: Average perimeter-to-area ratio of contours.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    complexities = [
        cv2.arcLength(contour, True) / (cv2.contourArea(contour) + 1e-5)
        for contour in contours
        if cv2.contourArea(contour) > 500
    ]
    return float(np.mean(complexities)) if complexities else 0.0


def object_density(image: np.ndarray) -> float:
    """
    Computes the density of objects in the image, which can help to identify crowded flowers.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: The density of objects per 100x100 pixels.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = image.shape[0] * image.shape[1]
    object_count = sum(1 for contour in contours if cv2.contourArea(contour) > 300)
    return object_count / (total_area / 10000)


def elongated_shapes_ratio(image: np.ndarray) -> float:
    """
    Identifies the proportion of elongated shapes (like stems) in the image.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        float: The ratio of elongated objects (ratio > 3 for elongated shapes).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elongated_count = 0
    total_count = 0

    for contour in contours:
        if cv2.contourArea(contour) > 300:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            total_count += 1
            if aspect_ratio > 3:
                elongated_count += 1

    return elongated_count / total_count if total_count > 0 else 0.0


feature_functions = {
    "contrast_flower": contrast_flower,
    "amount_of_yellow": amount_of_yellow,
    "amount_of_red": amount_of_red,
    "amount_of_green": amount_of_green,
    "avg_saturation": avg_saturation,
    "avg_brightness": avg_brightness,
    "circularity": circularity,
    "bud_shape": bud_shape,
    "count_objects": count_objects,
    "contour_complexity": contour_complexity,
    "object_density": object_density,
    "elongated_shapes_ratio": elongated_shapes_ratio,
}
