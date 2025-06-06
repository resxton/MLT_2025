import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Загрузка данных и подготовка (один раз при запуске)
@st.cache_resource
def load_data_and_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)
    
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_to_fix] = data[cols_to_fix].replace(0, np.nan)
    data.fillna(data.median(), inplace=True)
    
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

X, y, scaler = load_data_and_model()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Функция обучения трех моделей
def train_models(n_estimators, max_depth, min_samples_split):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth else 3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    return rf, lr, gb

# Интерфейс
st.title("👩‍⚕️ Прогнозирование диабета с 3 моделями")

st.sidebar.header("Параметры модели")
n_estimators = st.sidebar.slider(
    "Количество деревьев", 10, 200, 100, 10,
    help="Увеличивает точность, но может привести к переобучению"
)
max_depth = st.sidebar.selectbox(
    "Максимальная глубина", [None, 5, 10, 15, 20],
    help="Ограничение глубины деревьев"
)
min_samples_split = st.sidebar.slider(
    "Минимальное число образцов для разделения", 2, 10, 2,
    help="Контролирует переобучение"
)

# Кнопка переобучения
if st.sidebar.button("Переобучить все модели"):
    with st.spinner("Обучение моделей..."):
        rf_model, lr_model, gb_model = train_models(n_estimators, max_depth, min_samples_split)
        st.session_state['rf_model'] = rf_model
        st.session_state['lr_model'] = lr_model
        st.session_state['gb_model'] = gb_model
        
        # Оценка точности всех моделей
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
        lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
        gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
        
        st.sidebar.success("Модели успешно переобучены!")
        st.sidebar.metric("Random Forest точность", f"{rf_acc:.2%}")
        st.sidebar.metric("Logistic Regression точность", f"{lr_acc:.2%}")
        st.sidebar.metric("Gradient Boosting точность", f"{gb_acc:.2%}")

# Инициализация моделей при старте (если нет в сессии)
if 'rf_model' not in st.session_state:
    st.session_state['rf_model'] = RandomForestClassifier(random_state=42).fit(X_train, y_train)
if 'lr_model' not in st.session_state:
    st.session_state['lr_model'] = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
if 'gb_model' not in st.session_state:
    st.session_state['gb_model'] = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

# Ввод данных пациента
st.header("Данные пациента")
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.slider("Беременности", 0, 15, 0)
    glucose = st.slider("Глюкоза (mg/dL)", 50, 300, 90)
    blood_pressure = st.slider("Давление (mmHg)", 40, 160, 70)
    skin_thickness = st.slider("Толщина кожи (мм)", 5, 50, 20)
with col2:
    insulin = st.slider("Инсулин (μU/mL)", 2, 300, 16)
    bmi = st.slider("ИМТ (kg/m²)", 15.0, 50.0, 25.0, 0.1)
    diabetes_pedigree = st.slider("Генетический риск", 0.08, 2.5, 0.35, 0.01)
    age = st.slider("Возраст", 21, 90, 27)

# Кнопка расчёта вероятности
if st.button("Рассчитать вероятность"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    
    rf_proba = st.session_state['rf_model'].predict_proba(input_scaled)[0][1]
    lr_proba = st.session_state['lr_model'].predict_proba(input_scaled)[0][1]
    gb_proba = st.session_state['gb_model'].predict_proba(input_scaled)[0][1]
    
    st.subheader("Результаты моделей (вероятность диабета):")
    st.write(f"Random Forest: {rf_proba*100:.1f}%")
    st.write(f"Logistic Regression: {lr_proba*100:.1f}%")
    st.write(f"Gradient Boosting: {gb_proba*100:.1f}%")
    
    # Вывод предупреждений по самой высокой вероятности
    max_proba = max(rf_proba, lr_proba, gb_proba)
    if max_proba > 0.7:
        st.error("Высокий риск! Рекомендуется срочная консультация врача.")
    elif max_proba > 0.4:
        st.warning("Умеренный риск. Желательно пройти обследование.")
    else:
        st.success("Низкий риск.")
