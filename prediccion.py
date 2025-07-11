# ===============================
# 1. Importación de librerías
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


# ===============================
# 2. Configuración inicial
# ===============================
sns.set_theme(style="whitegrid")
plt.style.use('ggplot')
%config InlineBackend.figure_format = 'retina'


# ===============================
# 3. Carga y preparación de datos
# ===============================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/tesis_2025

    file_path = "data_resultado2.xlsx"
    df = pd.read_excel(file_path, engine='openpyxl')

    if df.empty:
        raise ValueError("El dataset está vacío")

    required_cols = ['ANO_EJE', 'PLIEGO_NOMBRE', 'MTO_DEVENGADO']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Faltan columnas requeridas en el dataset")

    df_anual = df.groupby(['ANO_EJE', 'PLIEGO_NOMBRE'])['MTO_DEVENGADO'].sum().reset_index()
    pliego_ejemplo = df_anual['PLIEGO_NOMBRE'].mode()[0]
    df_filtered = df_anual[df_anual['PLIEGO_NOMBRE'] == pliego_ejemplo].sort_values('ANO_EJE')

    if len(df_filtered) < 4:
        raise ValueError("Insuficientes datos temporales (mínimo 4 años requeridos)")

    data = df_filtered.set_index('ANO_EJE')[['MTO_DEVENGADO']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

except Exception as e:
    print(f"Error en preparación de datos: {e}")
    raise


# ===============================
# 4. Preparación para modelado
# ===============================
look_back = 1
train_size = max(2, int(len(scaled_data) * 0.7))
test_size = len(scaled_data) - train_size
if test_size < 1:
    train_size = len(scaled_data) - 1
    test_size = 1

train, test = scaled_data[:train_size], scaled_data[train_size:]

def create_sequences(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train, look_back)
X_test, y_test = create_sequences(test, look_back)

X_train = X_train.reshape((X_train.shape[0], look_back, 1))
X_test = X_test.reshape((X_test.shape[0], look_back, 1))


# ===============================
# 5. Modelo LSTM
# ===============================
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(look_back, 1)),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(0.001), loss='mse')
history = lstm_model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)

# Predicciones LSTM
train_predict = lstm_model.predict(X_train)
test_predict = lstm_model.predict(X_test)

# Inversión de escalado
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# ===============================
# 6. Modelo SARIMAX (ARIMA simplificado)
# ===============================
try:
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    arima_model = SARIMAX(train_data, order=(1, 1, 1))
    arima_fit = arima_model.fit(disp=False)
    arima_pred = arima_fit.get_forecast(steps=len(test_data)).predicted_mean

except Exception as e:
    print(f"Error en modelo ARIMA: {e}")
    arima_pred = np.zeros_like(test_data)


# ===============================
# 7. Evaluación de modelos
# ===============================
def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return rmse, mape

lstm_rmse, lstm_mape = evaluate_model(y_test_actual, test_predict)
arima_rmse, arima_mape = evaluate_model(test_data.values, arima_pred.values)


# ===============================
# 8. Visualización de resultados
# ===============================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Gráfico de predicciones
ax1.plot(data.index, data.values, label='Datos Reales', marker='o')
ax1.plot(data.index[look_back:train_size], train_predict, label='LSTM (Entrenamiento)')
ax1.plot(data.index[train_size+look_back:], test_predict, label='LSTM (Test)')
ax1.plot(arima_pred.index, arima_pred.values, label='ARIMA', linestyle='--')
ax1.set_title('Predicción de Ejecución Presupuestal')
ax1.set_ylabel('Monto (Millones S/.)')
ax1.legend()
ax1.grid(True)

# Gráfico de métricas
metrics = pd.DataFrame({
    'Modelo': ['LSTM', 'ARIMA'],
    'RMSE': [lstm_rmse, arima_rmse],
    'MAPE (%)': [lstm_mape, arima_mape]
})

ax2.bar(metrics['Modelo'], metrics['RMSE'], color='skyblue', label='RMSE')
ax2.bar(metrics['Modelo'], metrics['MAPE (%)'], color='orange', alpha=0.6, label='MAPE (%)')
ax2.set_title('Comparación de Métricas de Error')
ax2.set_ylabel('Valor')
ax2.legend()
ax2.grid(axis='y')

# Etiquetas de barra
for i, row in metrics.iterrows():
    ax2.text(i, row['RMSE'] + 0.5, f"{row['RMSE']:.2f}", ha='center')
    ax2.text(i, row['MAPE (%)'] + 0.5, f"{row['MAPE (%)']:.2f}%", ha='center')

plt.tight_layout()
plt.show()


# ===============================
# 9. Resultados tabulares
# ===============================
print("\nResultados Comparativos:")
display(
    metrics.style.format({
        'RMSE': '{:.2f}',
        'MAPE (%)': '{:.2f}%'
    }).set_caption("Métricas de Evaluación").background_gradient(cmap='Blues')
)
