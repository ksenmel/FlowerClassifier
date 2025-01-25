import cv2
import numpy as np


def contrast_flower(image: np.ndarray) -> float:
    """
    Вычисляет стандартное отклонение яркости изображения в оттенках серого. (ромашка или подсолнух)
    :param image:
    :return:
    """
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


def avg_saturation(image: np.ndarray) -> float:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Переводим изображение в HSV
    saturation = hsv_image[:, :, 1]  # Извлекаем канал насыщенности
    return float(np.mean(saturation))


def avg_brightness(image: np.ndarray) -> float:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Переводим изображение в HSV
    brightness = hsv_image[:, :, 2]  # Извлекаем канал яркости
    return float(np.mean(brightness))


def circularity(image: np.ndarray) -> float:
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
    res = 4 * np.pi * area / (perimeter**2)
    return res


def bud_shape(image: np.ndarray) -> float:
    """
    Вычисляет коэффициент удлиненности объекта, чтобы идентифицировать форму бутона цветов.
    Удлиненность определяется как отношение длины минимальной ограничивающей прямоугольной области
    к ее ширине.

    Args:
        image (np.ndarray): Входное изображение в формате BGR.

    Returns:
        float: Коэффициент удлиненности (больше 1 для вытянутых форм).
    """
    # Переводим изображение в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем пороговое преобразование для выделения объектов
    _, binary_image = cv2.threshold(
        gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0  # Если контуры не найдены, возвращаем 0

    # Выбираем самый крупный контур
    largest_contour = max(contours, key=cv2.contourArea)

    # Находим минимальный ограничивающий прямоугольник
    _, _, width, height = cv2.boundingRect(largest_contour)

    if height == 0:
        return 0.0  # Предотвращаем деление на ноль

    # Рассчитываем коэффициент удлиненности
    elongation_ratio = max(width, height) / min(width, height)

    return elongation_ratio


def count_objects(image: np.ndarray) -> int:
    """
    Считает количество отдельных объектов на изображении, которые могут быть одуванчиками или тюльпанами.

    Args:
        image (np.ndarray): Входное изображение в формате BGR.

    Returns:
        int: Количество обнаруженных объектов.
    """
    # Переводим изображение в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем размытие для уменьшения шума
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Бинаризация изображения для выделения объектов
    _, binary_image = cv2.threshold(
        blurred_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Находим контуры
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Считаем только крупные контуры, отбрасывая мелкие шумы
    object_count = sum(1 for contour in contours if cv2.contourArea(contour) > 500)

    return object_count


def contour_complexity(image: np.ndarray) -> float:
    """
    Вычисляет фрактальную сложность контура на изображении.
    Розы имеют сложные формы лепестков, которые можно измерить через фрактальную сложность контуров.

    Args:
        image (np.ndarray): Входное изображение в формате BGR.

    Returns:
        float: Среднее отношение периметра к площади для контуров.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    complexities = [
        cv2.arcLength(contour, True) / (cv2.contourArea(contour) + 1e-5)
        for contour in contours
        if cv2.contourArea(contour) > 500
    ]
    return np.mean(complexities) if complexities else 0.0


def object_density(image: np.ndarray) -> float:
    """
    Вычисляет плотность объектов на изображении, учитывая соотношение их количества к площади.

    Args:
        image (np.ndarray): Входное изображение в формате BGR.

    Returns:
        float: Плотность объектов (количество объектов на 100x100 пикселей).
    """
    # Перевод в оттенки серого и бинаризация
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    total_area = image.shape[0] * image.shape[1]

    # Считаем только крупные объекты
    object_count = sum(1 for contour in contours if cv2.contourArea(contour) > 300)

    # Рассчитываем плотность
    density = object_count / (total_area / 10000)  # На каждые 100x100 пикселей
    return density


def elongated_shapes_ratio(image: np.ndarray) -> float:
    """
    Определяет долю вытянутых объектов (стеблей) на изображении.

    Args:
        image (np.ndarray): Входное изображение в формате BGR.

    Returns:
        float: Доля вытянутых объектов относительно всех найденных объектов.
    """
    # Перевод в оттенки серого и бинаризация
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Поиск контуров
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    elongated_count = 0
    total_count = 0

    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Исключаем шум
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)  # Соотношение сторон объекта
            total_count += 1
            if aspect_ratio > 3:  # Условие для вытянутых форм
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
