import os
import json
import joblib

import pandas as pd
import streamlit as st


MODEL_PATH = os.path.join("output", "models", "best_model.joblib")
FEATURE_INFO_PATH = os.path.join("output", "models", "feature_info.json")


FEATURE_TRANSLATIONS = {
    "Warehouse_block": "Складской блок",
    "Mode_of_Shipment": "Способ доставки",
    "Customer_care_calls": "Количество звонков в поддержку",
    "Customer_rating": "Оценка клиента",
    "Cost_of_the_Product": "Стоимость товара",
    "Prior_purchases": "Количество прошлых покупок",
    "Product_importance": "Важность товара",
    "Gender": "Пол клиента",
    "Discount_offered": "Предложенная скидка",
    "Weight_in_gms": "Вес товара в граммах"
}

FEATURE_DESCRIPTIONS = {
    "Warehouse_block": "Блок склада, из которого отправляется заказ.",
    "Mode_of_Shipment": "Тип доставки: самолёт, корабль или наземная доставка.",
    "Customer_care_calls": "Сколько раз клиент обращался в службу поддержки.",
    "Customer_rating": "Оценка клиента по шкале от 1 до 5.",
    "Cost_of_the_Product": "Стоимость товара в условных денежных единицах.",
    "Prior_purchases": "Сколько покупок клиент совершал ранее.",
    "Product_importance": "Важность товара: низкая, средняя или высокая.",
    "Gender": "Пол клиента.",
    "Discount_offered": "Размер предложенной скидки в процентах.",
    "Weight_in_gms": "Вес товара в граммах."
}

NUMERIC_RANGES = {
    "Customer_care_calls": {
        "min": 2,
        "max": 7,
        "step": 1,
        "help": "Допустимый диапазон: от 2 до 7 звонков."
    },
    "Customer_rating": {
        "min": 1,
        "max": 5,
        "step": 1,
        "help": "Допустимый диапазон: от 1 до 5 баллов."
    },
    "Cost_of_the_Product": {
        "min": 96,
        "max": 310,
        "step": 1,
        "help": "Примерный диапазон в датасете: от 96 до 310."
    },
    "Prior_purchases": {
        "min": 2,
        "max": 10,
        "step": 1,
        "help": "Допустимый диапазон: от 2 до 10 прошлых покупок."
    },
    "Discount_offered": {
        "min": 1,
        "max": 65,
        "step": 1,
        "help": "Допустимый диапазон скидки: от 1% до 65%."
    },
    "Weight_in_gms": {
        "min": 1001,
        "max": 7846,
        "step": 1,
        "help": "Примерный диапазон веса: от 1001 до 7846 грамм."
    }
}

VALUE_TRANSLATIONS = {
    "Mode_of_Shipment": {
        "Flight": "Самолёт",
        "Ship": "Корабль",
        "Road": "Наземная доставка"
    },
    "Product_importance": {
        "low": "Низкая",
        "medium": "Средняя",
        "high": "Высокая"
    },
    "Gender": {
        "M": "Мужской",
        "F": "Женский"
    },
    "Warehouse_block": {
        "A": "Складской блок A",
        "B": "Складской блок B",
        "C": "Складской блок C",
        "D": "Складской блок D",
        "F": "Складской блок F"
    }
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_info():
    with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_russian_feature_name(feature):
    return FEATURE_TRANSLATIONS.get(feature, feature)


def get_feature_description(feature):
    return FEATURE_DESCRIPTIONS.get(feature, "")


def translate_value(feature, value):
    value_dict = VALUE_TRANSLATIONS.get(feature, {})
    return value_dict.get(value, str(value))


st.set_page_config(
    page_title="Прогноз доставки заказа",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Прогноз своевременности доставки заказа")

st.subheader(
    """
    Приложение прогнозирует, будет ли заказ доставлен вовремя, на основе характеристик товара, клиента и доставки.
    
    **Важно:** в исходном датасете целевая переменная имеет значения:
    
    - `0` — заказ доставлен вовремя;
    - `1` — заказ не доставлен вовремя.
    """
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Файл модели не найден. Сначала запустите основной проект командой: python project.py"
    )
    st.stop()

if not os.path.exists(FEATURE_INFO_PATH):
    st.error(
        "Файл feature_info.json не найден. Сначала запустите основной проект командой: python project.py"
    )
    st.stop()

model = load_model()
feature_info = load_feature_info()

numeric_features = feature_info["numeric_features"]
categorical_features = feature_info["categorical_features"]
categorical_values = feature_info["categorical_values"]
numeric_defaults = feature_info["numeric_defaults"]

st.subheader("Введите параметры заказа")

input_data = {}

st.markdown("## Числовые признаки")

for feature in numeric_features:
    russian_name = get_russian_feature_name(feature)
    description = get_feature_description(feature)
    default_value = float(numeric_defaults.get(feature, 0))
    range_info = NUMERIC_RANGES.get(feature)

    if range_info:
        min_value = float(range_info["min"])
        max_value = float(range_info["max"])
        step = float(range_info["step"])
        help_text = range_info["help"]

        if default_value < min_value:
            default_value = min_value
        if default_value > max_value:
            default_value = max_value

        st.markdown(f"**{russian_name}**")
        st.caption(description)
        st.caption(help_text)

        input_data[feature] = st.number_input(
            label=f"{russian_name} ({feature})",
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=step
        )
    else:
        st.markdown(f"**{russian_name}**")
        st.caption(description)

        input_data[feature] = st.number_input(
            label=f"{russian_name} ({feature})",
            value=default_value
        )

st.markdown("## Категориальные признаки")

for feature in categorical_features:
    russian_name = get_russian_feature_name(feature)
    description = get_feature_description(feature)
    values = categorical_values.get(feature, [])

    if len(values) == 0:
        st.error(f"Нет доступных значений для признака {feature}")
        st.stop()

    display_options = []

    for value in values:
        translated = translate_value(feature, value)
        display_options.append(f"{translated} ({value})")

    st.markdown(f"**{russian_name}**")
    st.caption(description)

    selected_display_value = st.selectbox(
        label=f"{russian_name} ({feature})",
        options=display_options
    )

    selected_original_value = selected_display_value.split("(")[-1].replace(")", "")
    input_data[feature] = selected_original_value

st.markdown("---")

if st.button("Сделать прогноз"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.subheader("Результат прогноза")

    if prediction == 0:
        st.success("✅ Заказ, скорее всего, будет доставлен вовремя.")
    else:
        st.error("⚠️ Заказ, скорее всего, НЕ будет доставлен вовремя.")

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            st.markdown("### Вероятности по классам")

            readable_classes = []

            for class_value in classes:
                if class_value == 0:
                    readable_classes.append("0 — доставлен вовремя")
                elif class_value == 1:
                    readable_classes.append("1 — не доставлен вовремя")
                else:
                    readable_classes.append(str(class_value))

            proba_df = pd.DataFrame({
                "Класс": readable_classes,
                "Вероятность": probabilities
            })

            st.dataframe(proba_df)

            if 0 in classes:
                class_index = list(classes).index(0)
                st.info(
                    f"Вероятность своевременной доставки: "
                    f"{probabilities[class_index]:.2%}"
                )

            if 1 in classes:
                class_index = list(classes).index(1)
                st.info(
                    f"Вероятность задержки доставки: "
                    f"{probabilities[class_index]:.2%}"
                )

        except Exception:
            st.info("Модель не смогла вывести вероятности классов.")

