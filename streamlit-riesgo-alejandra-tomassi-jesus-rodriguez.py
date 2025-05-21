import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from scipy import stats
from PIL import Image
from scipy.stats import yeojohnson, yeojohnson_normmax, yeojohnson_llf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def inv_yeojohnson(y, lmbda):
    y = np.asarray(y)
    if lmbda == 0:
        return np.exp(y) - 1
    else:
        return np.where(y >= 0,
                        np.power(y * lmbda + 1, 1 / lmbda) - 1,
                        -np.power(-y * (2 - lmbda) + 1, 1 / (2 - lmbda)) + 1)

# ────────────────────────────────────────────────────────────────
# Configuración y semillas
# ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
st.set_page_config(page_title="Caligus Risk Dashboard", layout="wide")

# ────────────────────────────────────────────────────────────────
# 1️⃣ Carga de datos
# ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Sube CSV con columnas: ho, juvenil, temperatura(°c)_mean, salinidad(ppt)_mean, tipo_*, semana_*",
    type="csv",
)
if not uploaded:
    st.stop()
df = pd.read_csv(uploaded)

# ────────────────────────────────────────────────────────────────
# 2️⃣ Definición de columnas
# ────────────────────────────────────────────────────────────────
base_cols   = ["temperatura(°c)_mean", "salinidad(ppt)_mean"]
tipo_cols   = [c for c in df.columns if c.startswith("tipo_")]
semana_cols = [c for c in df.columns if c.startswith("semana_")]
cols_lstm = ["juvenil"] + base_cols + semana_cols + tipo_cols
required    = ["ho", "juvenil"] + base_cols
missing     = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Faltan columnas obligatorias: {missing}")
    st.stop()

yj_lambda = df["lambda"].values[0] if "lambda" in df.columns else None
variable_objetivo = "juvenil"
variable_exogena = list(["temperatura(°c)_mean", "salinidad(ppt)_mean", ] + [col for col in df.columns if col.startswith("semana_")] + [col for col in df.columns if col.startswith("tipo_")]) 

# ────────────────────────────────────────────────────────────────
# 3️⃣ Preprocesado: Yeo-Johnson + scaler único y secuencia lag=1
# ────────────────────────────────────────────────────────────────


def generar_metricas(model_obj, X_test, y_test, scaler, variable_exogena):
    # 1) Predicción sobre el conjunto de prueba (escala transformada)
    predicciones_S = model_obj.predict(X_test)

    # 2) Crear matrices para desnormalizar
    pred_ajust     = np.zeros((predicciones_S.shape[0], len(variable_exogena) + 1))
    test_ajust     = np.zeros((y_test.shape[0],      len(variable_exogena) + 1))

    # 3) Rellenar la primera columna con predicción y verdadero
    pred_ajust[:, 0] = predicciones_S.ravel()
    test_ajust[:, 0] = y_test.ravel()

    # 4) Desnormalizar todo el array
    pred_R = scaler.inverse_transform(pred_ajust)[:, 0]
    true_R = scaler.inverse_transform(test_ajust)[:, 0]

    # -------------------- EVALUACIÓN -------------------- #

    mae_val   = mean_absolute_error(true_R, pred_R)
    rmse_val  = np.sqrt(mean_squared_error(true_R, pred_R))
    r2_val    = r2_score(true_R, pred_R)
    nz        = true_R != 0
    mape_val  = np.mean(np.abs((true_R[nz] - pred_R[nz]) / true_R[nz])) * 100
    rmspe_val = np.sqrt(np.mean(((true_R[nz] - pred_R[nz]) / true_R[nz])**2)) * 100
    return pred_R, true_R, mae_val, rmse_val, r2_val, mape_val, rmspe_val

def preparar_set(df, test_size=0.2, transformacion="boxcox", scaler=None, model_type="sequential"):
    """
    Prepara el set de datos para:
    - "sequential": sliding window (lags de ho y exógenas)
    - "regressor": solo exógenas, evitando data leakage
    Devuelve X_train, X_test, y_train, y_test, objetivo, X, y, scaler
    """
    # Definir columnas: target y exógenas
    target_col = f"{variable_objetivo}_{transformacion}" if transformacion else variable_objetivo
    expected_columns = [target_col] + variable_exogena

    df = df.copy()

    # Rellenar columnas faltantes
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Extract matrix and scale
    objetivo = df[expected_columns].values.astype(float)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0,1))
        data_scaled = scaler.fit_transform(objetivo)
    else:
        data_scaled = scaler.transform(objetivo)

    # Guardar completo escalado
    objetivo_scaled = data_scaled.copy()

    # Construir X,y según tipo
    if model_type == "sequential":
        X_list, y_list = [], []
        time_step = 1
        for i in range(len(data_scaled) - time_step):
            X_list.append(data_scaled[i:i+time_step, :])
            y_list.append(data_scaled[i+time_step, 0])
        X = np.array(X_list)
        y = np.array(y_list)
    elif model_type == "regressor":
        # Evitar data leak: X solo exógenas, y es ho
        y = data_scaled[:, 0]
        X = data_scaled[:, 1:]
    else:
        raise ValueError("model_type debe ser 'sequential' o 'regressor'.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )

    # Rounding idéntico al original
    X_train = np.round(X_train, 2)
    X_test  = np.round(X_test, 2)
    y_train = np.round(y_train, 2)
    y_test  = np.round(y_test, 2)
    objetivo = np.round(objetivo, 2)
    X = np.round(X, 2)
    y = np.round(y, 2)

    return X_train, X_test, y_train, y_test, objetivo, X, y, scaler

X_tr, X_te, y_tr, y_te, objetivo, X, y, scaler = preparar_set(df, transformacion="yeojohnson", scaler=None, model_type="sequential")

# ────────────────────────────────────────────────────────────────
# 4️⃣ Entrenamiento automático de la LSTM
# ────────────────────────────────────────────────────────────────

# -------------------- MODELAMIENTO -------------------- #

# Crear el modelo LSTM
model_lstm = Sequential()

# Capa oculta (LSTM)
model_lstm.add(LSTM(
    units=50,
    activation='sigmoid', # opciones 'relu', 'linear', 'tanh', 'sigmoid'
    input_shape=(X_tr.shape[1], X_tr.shape[2]),
    recurrent_regularizer=l1_l2(l1=0.01, l2=0.01)
))

# Capa de salida
model_lstm.add(Dense(units=1))

# -------------------- COMPILACIÓN -------------------- #

model_lstm.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='mean_squared_error'
)

early_stopping = EarlyStopping(
    monitor='val_loss',         # se fija en la pérdida de validación
    patience=15,                 # cuántas épocas esperar antes de detener si no hay mejora
    restore_best_weights=True   # al terminar, vuelve a la época con menor val_loss
)

# -------------------- ENTRENAMIENTO -------------------- #

entrenamiento_LSTM = model_lstm.fit(
    X_tr, y_tr,
    epochs=150,
    batch_size=64,
    validation_data=(X_te, y_te),
    callbacks=[early_stopping]  # se agrega el callback
)

# ────────────────────────────────────────────────────────────────
# 5️⃣ Entrenamiento automático del Random Forest
# ────────────────────────────────────────────────────────────────
variable_exogena_clasificador = list(["temperatura(°c)_mean","salinidad(ppt)_mean", "juvenil"] + [col for col in df.columns if col.startswith("tipo_")] + [col for col in df.columns if col.startswith("semana_")]) 

X_rf = df[[col for col in variable_exogena_clasificador if not col.startswith('semana_')]].values
y_rf = df["ho"].values
X_trf, X_tef, y_trf, y_tef = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=SEED
)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
clf_rf.fit(X_trf, y_trf)

# ────────────────────────────────────────────────────────────────
# 6️⃣ Cabecera con logo + título
# ────────────────────────────────────────────────────────────────
col_img, col_title = st.columns([1, 9])
try:
    logo = Image.open("caligus.jpeg")
    col_img.image(logo, width=80)
except:
    pass
col_title.title("Predicción Juvenil + Clasificación de Riesgo ho")
col_title.markdown("Entrena modelos y clasifica el nivel de riesgo operativo de Caligus")
col_title.markdown("El modelo usado es un Random Forest entrenado con datos de juveniles y condiciones ambientales")

# ────────────────────────────────────────────────────────────────
# 7️⃣ Estado de entrenamiento
# ────────────────────────────────────────────────────────────────
# Streamlit's st.expander does not support a font_size argument.
# To change font size, use HTML/CSS in st.markdown inside the expander.

with st.expander("Estado de entrenamiento y métricas....más", expanded=False):
    st.markdown("<span style='font-size:18px;'>- LSTM (juveniles): entrenada</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:18px;'>- Random Forest (riesgo HO): entrenado</span>", unsafe_allow_html=True)

    # Reporte de métricas LSTM
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_pred_lstm = model_lstm.predict(X_te).flatten()
    y_test_lstm = scaler.inverse_transform(np.c_[y_te, np.zeros((len(y_te), len(cols_lstm)-1))])[:,0]
    y_pred_lstm_inv = scaler.inverse_transform(np.c_[y_pred_lstm, np.zeros((len(y_pred_lstm), len(cols_lstm)-1))])[:,0]
    y_test_lstm = inv_yeojohnson(y_test_lstm, yj_lambda)
    y_pred_lstm_inv = inv_yeojohnson(y_pred_lstm_inv, yj_lambda)

    pred_LSTM, true_LSTM, mae_LSTM, rmse_LSTM, r2_LSTM, mape_LSTM, rmspe_LSTM = generar_metricas(model_lstm, X_te, y_te, scaler, variable_exogena)

    st.markdown("<span style='font-size:16px;'><b>Métricas LSTM (test)</b></span>", unsafe_allow_html=True)
    st.write(f"MAE: {mae_LSTM:.2f}")
    st.write(f"RMSE: {rmse_LSTM:.2f}")
    st.write(f"R2: {r2_LSTM:.2f}")


    # Reporte de métricas Random Forest
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

    rf_pred = clf_rf.predict(X_tef)

    # Calcular métricas de clasificación
    rf_accuracy = accuracy_score(y_tef, rf_pred)
    rf_precision = precision_score(y_tef, rf_pred, average='weighted')
    rf_recall = recall_score(y_tef, rf_pred, average='weighted')
    rf_f1 = f1_score(y_tef, rf_pred, average='weighted')


    report = classification_report(y_tef, rf_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T
    st.markdown("<span style='font-size:16px;'><b>Métricas Random Forest (test)</b></span>", unsafe_allow_html=True)
    st.dataframe(df_report.style.format({
        "precision": "{:.2f}",
        "recall":    "{:.2f}",
        "f1-score":  "{:.2f}",
        "support":   "{:.0f}"
    }))

# ────────────────────────────────────────────────────────────────
# 8️⃣ Clasificación interactiva
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Clasificación interactiva")

# Inputs dinámicos
temp        = st.number_input("Temperatura (°C)", 
    value = float(df["temperatura(°c)_mean"].median()),
    min_value=df["temperatura(°c)_mean"].min(),
    max_value=df["temperatura(°c)_mean"].max() + 5)
sal         = st.number_input("Salinidad (ppt)",  
    value = float(df["salinidad(ppt)_mean"].median()),
    min_value=df["salinidad(ppt)_mean"].min(), 
    max_value=df["salinidad(ppt)_mean"].max() + 10)
juvenile_t1 = st.number_input(
    "Juvenil t-1",
    value=float(df["juvenil"].median()),
    min_value=0.0,
    max_value=float(df["juvenil"].max()) * 10
)
semana      = st.selectbox("Semana", semana_cols)
tipo        = st.selectbox("Tipo de Tratamiento", tipo_cols)

# Ajuste: usar solo un scaler para todo
scaler_full = MinMaxScaler().fit(df[cols_lstm])

# 8.1 Predicción automática de juveniles
# Aplicar Yeo-Johnson a juvenil t-1 antes de escalar
juvenile_t1_yj = yeojohnson(juvenile_t1, lmbda=yj_lambda)
feat_lstm = [juvenile_t1_yj, temp, sal] + [1 if s==semana else 0 for s in semana_cols] + [1 if s==tipo else 0 for s in tipo_cols]
input_df = pd.DataFrame([feat_lstm], columns=cols_lstm)
input_scaled = scaler.transform(input_df)
X_input = input_scaled.reshape(1, 1, -1)
y_lstm = model_lstm.predict(X_input)[0, 0]
# Inversa solo para juvenil: primero desescalar, luego inverse Yeo-Johnson
juvenile_pred_scaled = scaler.inverse_transform([[y_lstm] + [0]*(len(cols_lstm)-1)])[0, 0]
# Inversa Yeo-Johnson
# scipy no tiene inv_yeojohnson, pero se puede usar la fórmula:
juvenile_pred = inv_yeojohnson(juvenile_pred_scaled, yj_lambda)
juvenile_pred = max(juvenile_pred, 0)  # evitar negativos

st.markdown("**Juveniles esperados para la siguiente semana**")
st.markdown(
    f"<div style='background:#f0f0f0; padding:8px;'>"
    f"<span style='color:green; font-weight:bold; font-size:24px;'>{juvenile_pred:.2f}</span>"
    "</div>",
    unsafe_allow_html=True,
)

# 8.2 Botón para clasificar riesgo
st.markdown("\n")
if st.button("Clasificar Riesgo"):
    rf_features = [juvenile_pred, temp, sal] + [1 if s==tipo else 0 for s in tipo_cols]
    rf_vec      = np.array(rf_features).reshape(1, -1)
    risk        = clf_rf.predict(rf_vec)[0]

    # recuadro de resultado
    color_map = {"Bajo":"#2ca02c", "Medio":"#ff7f0e", "Alto":"#d62728"}
    col        = color_map.get(risk, "black")
    msg        = f"El riesgo de infestación de ho para la siguiente seamana es: {risk}"
    st.markdown(
        f"<div style='border:2px solid {col}; padding:12px; border-radius:6px;'>"
        f"{msg}"
        "</div>",
        unsafe_allow_html=True,
    )
