import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


actions = [119, None]
# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table_final.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
X = np.array(X)


y = np.array(y)
# --- Definir la red neuronal ---
model = keras.Sequential([
    keras.layers.Input(shape=(5,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(actions), activation='linear')
])


model.compile(optimizer='adam', loss='mse')

# --- Entrenar la red neuronal ---
# COMPLETAR: Ajustar hiperparámetros según sea necesario

early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

history = model.fit(X, y, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stop])


# --- Mostrar resultados del entrenamiento ---
train_loss = model.evaluate(X, y, verbose=0)
print(f"Error cuadrático medio final: {train_loss:.4f}")
# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas
print(model.predict(X[10:11], verbose=0),y[10])

model.save('flappy_q_nn_model.keras')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
