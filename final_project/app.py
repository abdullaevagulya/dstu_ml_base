"""Каркас Streamlit-приложения. Допишите три функции с TODO."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Регрессия": ROOT / "models" / "price_model.joblib",
    "Классификация": ROOT / "models" / "class_model.joblib",
}


def load_artifact(path):
    """Верните словарь с model, features, preparation и threshold для классификатора."""
    # TODO: прочитайте свой файл через joblib.load(path).
    # Проверьте структуру словаря; если файл отсутствует, объясните, что нужно
    # сначала выполнить ноутбук. Не обучайте модель внутри приложения.
    raise NotImplementedError("Допишите load_artifact: загрузка модели из файла.")


def prepare_input(frame, features, preparation):
    """Верните числовую таблицу в порядке столбцов обучения или поднимите ValueError."""
    # TODO: проверьте пустой файл, наличие столбцов, числовые значения,
    # положительную площадь и непротиворечивость этажей.
    # Для классификации в features входит общая цена price_rub в рублях.
    # Проверьте, что цена положительна и заполнена.
    # Выберите features и повторите подготовку из ноутбука: заполнение
    # пропусков, кодирование и тот же порядок столбцов. Используйте значения
    # из preparation: медианы, columns и, если нужно, сохранённый scaler.
    # Не вычисляйте их заново по загруженным квартирам.
    raise NotImplementedError("Допишите prepare_input: проверка входных данных.")


def predict_table(frame, artifact, task):
    """Верните копию исходной таблицы с добавленными прогнозами."""
    # TODO: получите X через prepare_input(frame, artifact["features"], artifact["preparation"]).
    # Регрессия: artifact["model"].predict(X) -> столбец prediction_rub.
    # Классификация: predict_proba(X), найдите столбец класса 1 через classes_.
    # Добавьте probability и predicted_class, используя сохранённый threshold.
    # Правило: p >= threshold; не округляйте p перед вычислением класса.
    # Проверьте число прогнозов, конечность значений, диапазон вероятностей.
    raise NotImplementedError("Допишите predict_table: получение прогнозов.")


def main():
    st.set_page_config(page_title="Квартиры: две модели")
    st.title("Прогноз для квартиры")
    task = st.selectbox("Выберите задачу", list(MODEL_FILES))
    st.caption("Для классификации укажите общую цену квартиры в рублях: price_rub.")
    st.info("Стартовый каркас: сначала обучите модели и допишите функции с TODO.")
    # TODO: опишите входные поля и единицы; добавьте скачивание примера CSV.
    uploaded = st.file_uploader("CSV с параметрами квартир", type=["csv"])
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
        st.dataframe(frame)
        if st.button("Получить прогноз"):
            artifact = load_artifact(MODEL_FILES[task])
            result = predict_table(frame, artifact, task)
            st.dataframe(result)
            st.download_button(
                "Скачать прогнозы", result.to_csv(index=False).encode("utf-8-sig"),
                file_name="predictions.csv", mime="text/csv",
            )
    except NotImplementedError as error:
        st.info(str(error))
    except FileNotFoundError:
        st.error("Файл модели не найден. Сначала выполните нужный ноутбук.")
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        st.error(f"Проверьте файл и формат данных: {error}")


if __name__ == "__main__":
    main()
