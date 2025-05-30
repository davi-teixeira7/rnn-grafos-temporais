# Importação de bibliotecas
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
import os # Importar o módulo os

# Para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Obter o diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================
# 1. Carregar a base de dados
# ============================================

# Definir o nome do arquivo de dados

# data_filename = 'CollegeMsg.txt'
# data_filename = 'email-Eu-core-temporal-Dept2.txt'
# data_filename = 'email-Eu-core-temporal-Dept3.txt'
data_filename = 'email-Eu-core-temporal-Dept4.txt'
# data_filename = 'email-Eu-core-temporal.txt'
filepath = os.path.join(script_dir, 'data', data_filename)

try:
    df = pd.read_csv(filepath, sep=' ', header=None, names=['SRC', 'TGT', 'TS'])
    print(f"Arquivo de dados '{filepath}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo {filepath} não foi encontrado.")
    print(f"Verifique se o arquivo '{data_filename}' existe no subdiretório 'data' dentro de '{script_dir}'.")
    exit() # Termina o script se o arquivo não for encontrado
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo de dados: {e}")
    exit()

# ============================================
# 2. Converter timestamp para dia
# ============================================
df['day'] = (df['TS'] // (24*60*60)).astype(int)  # segundos -> dias

# ============================================
# 3. Criar snapshots diários e calcular métricas
# ============================================

metrics = []

for day, group in df.groupby('day'):
    G = nx.Graph()
    edges = list(zip(group['SRC'], group['TGT']))
    G.add_edges_from(edges)
    
    # Grau médio
    if len(G.nodes) > 0:
        avg_degree = np.mean([d for n, d in G.degree()])
        clustering = nx.average_clustering(G)
    else:
        avg_degree = 0
        clustering = 0
    
    metrics.append({'day': day, 'avg_degree': avg_degree, 'clustering': clustering})

metrics_df = pd.DataFrame(metrics)

if metrics_df.empty:
    print("Nenhuma métrica foi calculada. Verifique os dados de entrada.")
    exit()

# ============================================
# 4. Criar séries temporais
# ============================================

# Exemplo: usar grau médio como alvo
series = metrics_df['avg_degree'].values

# Criar janelas de sequência
def create_sequences(data, seq_length):
    X, y = [], []
    if len(data) <= seq_length: # Verifica se há dados suficientes para criar sequências
        print(f"Não há dados suficientes (comprimento: {len(data)}) para criar sequências com seq_length={seq_length}")
        return np.array(X), np.array(y) # Retorna arrays vazios
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(series, seq_length)

if X.shape[0] == 0: # Verifica se alguma sequência foi criada
    print("Nenhuma sequência de treinamento foi criada. Verifique 'seq_length' e os dados da série temporal.")
    exit()

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    print("Divisão treino/teste resultou em conjuntos vazios. Verifique os dados e test_size.")
    exit()

# ============================================
# 5. Modelos de redes neurais
# ============================================

# ------ MLP ------
mlp = Sequential([
    Dense(128, activation='relu', input_shape=(seq_length,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])

mlp.compile(optimizer='adam', loss='mae')
print("\nTreinando MLP...")
mlp.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Avaliação
y_pred = mlp.predict(X_test)
print('MLP MAE:', mean_absolute_error(y_test, y_pred))

# ------ LSTM ------
# LSTM precisa de entrada 3D: (batch_size, timesteps, features)
X_train_lstm = X_train[..., np.newaxis]
X_test_lstm = X_test[..., np.newaxis]

lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(128),
    Dense(1)
])

lstm.compile(optimizer='adam', loss='mae')
print("\nTreinando LSTM...")
lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=16, verbose=1)

y_pred_lstm = lstm.predict(X_test_lstm)
print('LSTM MAE:', mean_absolute_error(y_test, y_pred_lstm))

# ------ CNN ------
cnn = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

cnn.compile(optimizer='adam', loss='mae')
print("\nTreinando CNN...")
cnn.fit(X_train_lstm, y_train, epochs=50, batch_size=16, verbose=1)

y_pred_cnn = cnn.predict(X_test_lstm)
print('CNN MAE:', mean_absolute_error(y_test, y_pred_cnn))

# ============================================
# 6. Visualizar resultados (Salvar em arquivo)
# ============================================

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Real', color='blue', linestyle='--')
plt.plot(y_pred, label='MLP', color='red', alpha=0.7)
plt.plot(y_pred_lstm, label='LSTM', color='green', alpha=0.7)
plt.plot(y_pred_cnn, label='CNN', color='purple', alpha=0.7)
plt.legend()
plt.title('Comparação de Modelos na Predição de Grau Médio')
plt.xlabel('Amostras de Teste')
plt.ylabel('Grau Médio')
plt.grid(True)


images_dir_path = os.path.join(script_dir, 'images')
data_filename_base = os.path.splitext(data_filename)[0]
output_plot_path = os.path.join(images_dir_path, f'comparacao_modelos_predicao-{data_filename_base}.png')


try:
    plt.savefig(output_plot_path)
    print(f"\nGráfico salvo com sucesso em: {output_plot_path}")
except Exception as e:
    print(f"\nOcorreu um erro ao salvar o gráfico: {e}")


