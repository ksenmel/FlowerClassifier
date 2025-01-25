import numpy as np
import pytest
from sklearn.metrics import f1_score
from classification.classifier import Classifier
from classification.features import amount_of_yellow


# evaluate test
def test_evaluate():
    # Создаём классификатор
    classifier = Classifier()

    # Мнимые предсказания и метки
    y_test = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    # Рассчитываем F1-скор
    expected_f1 = f1_score(y_test, y_pred, average="weighted")
    assert classifier.evaluate(y_test, y_pred) == pytest.approx(expected_f1, rel=1e-2)
