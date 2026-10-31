# streamlit создаёт веб-приложения на python: кнопки, загрузку файлов и таблицы
# документация: https://docs.streamlit.io/
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# __file__ это app.py, resolve() получает полный путь, parent берёт его папку
LESSON_DIR = Path(__file__).resolve().parent


# кеш сохраняет модель в памяти, чтобы не читать файл при каждом действии
@st.cache_resource
def load_model(filename):
    # собираем путь к модели, rb открывает файл для чтения в двоичном виде
    with open(LESSON_DIR / "models" / filename, "rb") as file:
        # восстанавливаем обученную модель из pickle
        return pickle.load(file)


def check_data(data, features):
    # выбираем нужные столбцы в том же порядке, что при обучении
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError("не хватает столбцов: " + ", ".join(missing))
    if data.empty:
        raise ValueError("в файле нет квартир")
    X = data[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("заполните пропуски и уберите бесконечные значения")

    # проверяем числовые поля и готовые бинарные признаки
    numeric = ["Всего этажей в доме", "Общая площадь", "Этаж", "rooms", "passenger_lift_count"]
    binary = [column for column in features if column not in numeric]
    if not X[binary].isin([0, 1]).all().all():
        raise ValueError("бинарные признаки должны содержать только 0 или 1")
    if (X[["Общая площадь", "Всего этажей в доме", "Этаж"]] <= 0).any().any():
        raise ValueError("площадь, этаж и число этажей должны быть больше нуля")
    if (X["rooms"] < 0).any() or (X["passenger_lift_count"] < -1).any():
        raise ValueError("комнаты должны быть от 0, лифты от 0 или -1, если неизвестно")
    counts = ["Всего этажей в доме", "Этаж", "rooms", "passenger_lift_count"]
    if (X[counts] % 1 != 0).any().any():
        raise ValueError("число этажей, этаж, комнаты и лифты должны быть целыми")
    if (X["Этаж"] > X["Всего этажей в доме"]).any():
        raise ValueError("этаж квартиры не может быть выше числа этажей в доме")
    for prefix in ["city_", "Тип дома_"]:
        columns = [column for column in features if column.startswith(prefix)]
        if (X[columns].sum(axis=1) > 1).any():
            raise ValueError("для города и типа дома можно выбрать не больше одного признака")
    return X


st.set_page_config(page_title="Оценка квартиры", page_icon="🏠")
st.title("Оценка стоимости квартиры")
st.caption(
    "Streamlit превращает код Python в веб-приложение с кнопками и таблицами. "
    "[Документация Streamlit](https://docs.streamlit.io/)."
)
st.subheader(
    "Две модели: предсказание цены квартиры и определение по её параметрам, "
    "относится ли она к квартирам с ценой выше рынка."
)
# переключатель слева выбирает задачу и нужную модель
mode = st.sidebar.radio(
    "Выберите задачу",
    ["Предсказать цену", "Проверить, выше ли рынка"],
)
st.caption(
    "Учебные модели на данных курса. Классификатор использует параметры квартиры, "
    "без её цены. Его классы заданы учебными городскими ориентирами из занятия 4, "
    "поэтому результат не является проверкой конкретной цены на текущем рынке."
)

filename = "price_model.pkl" if mode == "Предсказать цену" else "class_model.pkl"
model = load_model(filename)
features = list(model.feature_names_in_)

# показываем формат входных данных
st.write("Загрузите CSV или Excel (.xlsx): одна строка соответствует одной квартире.")
with st.expander("Какие поля нужны в файле"):
    st.markdown("#### Параметры квартиры")
    st.caption("Заголовки должны совпадать с названиями ниже. Все 19 полей обязательны.")
    descriptions = {
        "Тип рынка": "0 или 1, код рынка из исходной таблицы",
        "Всего этажей в доме": "количество этажей, целое число больше 0",
        "Общая площадь": "площадь в м², число больше 0",
        "Этаж": "этаж квартиры, от 1 до числа этажей в доме",
        "rooms": "количество комнат, 0 для студии",
        "has_Парковка": "1 есть парковка, 0 нет",
        "has_Балкон или лоджия": "1 есть балкон или лоджия, 0 нет",
        "has_Спортивная площадка": "1 есть спортивная площадка, 0 нет",
        "passenger_lift_count": "число пассажирских лифтов, -1 если неизвестно",
    }
    for column in features:
        if column.startswith("city_"):
            descriptions[column] = "1 если город " + column.removeprefix("city_") + ", иначе 0"
        elif column.startswith("Тип дома_"):
            descriptions[column] = "1 если дом " + column.removeprefix("Тип дома_") + ", иначе 0"
    st.table(pd.DataFrame({
        "Название столбца": features,
        "Как заполнить": [descriptions[column] for column in features],
    }).set_index("Название столбца"))
    st.info(
        "Город: выберите не больше одного city_. Для Дербента все city_ равны 0.\n\n"
        "Тип дома: выберите не больше одного признака. Для базового типа все равны 0."
    )
    st.caption("Лишние столбцы, включая цену и правильный класс, игнорируются.")
uploaded = st.file_uploader("Файл с параметрами квартир", type=["csv", "xlsx"])

if uploaded is not None:
    try:
        # читаем таблицу и проверяем её до передачи модели
        if uploaded.name.lower().endswith(".xlsx"):
            data = pd.read_excel(uploaded, engine="openpyxl")
        else:
            data = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
        X = check_data(data, features)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"Не удалось прочитать данные: {error}")
        st.stop()
    st.dataframe(X, hide_index=True)

    if st.button("Получить результат", type="primary"):
        predictions = model.predict(X)
        result = data.copy()
        if mode == "Предсказать цену":
            result["стоимость, руб."] = predictions.round(2)
            if len(result) == 1:
                st.success(f"Стоимость вашей квартиры: {predictions[0]:,.0f} руб.")
            else:
                st.success("Стоимость ваших квартир указана в таблице, в рублях.")
        else:
            # каждому числовому классу добавляем понятный текст
            result["класс"] = predictions.astype(int)
            result["ответ"] = [
                "ваша квартира выше рынка" if value == 1
                else "ваша квартира не выше рынка"
                for value in predictions
            ]
            if len(result) == 1:
                st.success(f"{int(predictions[0])}: {result['ответ'].iloc[0]}")
            else:
                st.success("Классы ваших квартир: 0 означает не выше рынка, 1 означает выше рынка.")
        st.dataframe(result, hide_index=True)
        st.download_button(
            "Скачать результат CSV",
            result.to_csv(index=False).encode("utf-8-sig"),
            file_name="predictions.csv",
            mime="text/csv",
        )
