# models_melhorado.py

import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

def criar_mlp(seq_length: int, regularization: float = 0.01) -> tf.keras.Model:
    """
    Cria Multilayer Perceptron para predição de séries temporais.
    
    Características:
    - Trata sequência como vetor simples (sem memória temporal explícita)
    - 3 camadas ocultas com 128 neurônios cada
    - Dropout para regularização
    - Ativação ReLU nas camadas ocultas, linear na saída
    
    Args:
        seq_length: Comprimento da sequência de entrada
        regularization: Fator de regularização L2
    
    Returns:
        Modelo MLP compilado
    """
    model = Sequential([
        Input(shape=(seq_length,)),
        
        # Primeira camada oculta
        Dense(128, activation='relu', 
              kernel_regularizer=regularizers.l2(regularization)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Segunda camada oculta
        Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(regularization)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Terceira camada oculta
        Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(regularization)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Camada de saída
        Dense(1, activation='linear')
    ], name='MLP_Model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse', 'mae']
    )
    
    logger.info(f"MLP criado: {model.count_params()} parâmetros")
    return model

def criar_lstm(seq_length: int, regularization: float = 0.01) -> tf.keras.Model:
    """
    Cria rede LSTM para predição de séries temporais.
    
    Características:
    - Processa sequência temporalmente (mantém estados internos)
    - 2 camadas LSTM com 128 unidades cada
    - Estados internos: h(t) memória curto prazo, c(t) memória longo prazo
    - Dropout entre camadas para regularização
    
    Funcionamento LSTM:
    1. Forget Gate: decide que informações descartar do estado da célula
    2. Input Gate: decide que valores atualizar no estado da célula
    3. Output Gate: decide que partes do estado da célula usar na saída
    
    Args:
        seq_length: Comprimento da sequência de entrada
        regularization: Fator de regularização L2
    
    Returns:
        Modelo LSTM compilado
    """
    model = Sequential([
        Input(shape=(seq_length, 1)),
        
        # Primeira camada LSTM (retorna sequências completas)
        LSTM(128, 
             return_sequences=True,
             dropout=0.3,
             recurrent_dropout=0.2,
             kernel_regularizer=regularizers.l2(regularization)),
        
        # Segunda camada LSTM (retorna apenas último output)
        LSTM(128,
             dropout=0.3,
             recurrent_dropout=0.2,
             kernel_regularizer=regularizers.l2(regularization)),
        
        # Camada densa final
        Dense(50, activation='relu',
              kernel_regularizer=regularizers.l2(regularization)),
        Dropout(0.2),
        
        # Camada de saída
        Dense(1, activation='linear')
    ], name='LSTM_Model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse', 'mae']
    )
    
    logger.info(f"LSTM criado: {model.count_params()} parâmetros")
    return model

def criar_cnn(seq_length: int, regularization: float = 0.01) -> tf.keras.Model:
    """
    Cria rede CNN 1D para predição de séries temporais.
    
    Características:
    - Aplica convolução 1D para detectar padrões locais na sequência
    - Filtros convolucionais deslizam pela série temporal
    - MaxPooling reduz dimensionalidade mantendo características importantes
    - Adequada para detectar padrões repetitivos/cíclicos
    
    Funcionamento:
    1. Conv1D: Filtros detectam padrões locais (ex: tendências, picos)
    2. MaxPooling: Reduz dimensão, mantém características mais relevantes
    3. Flatten: Transforma saída 2D em vetor 1D
    4. Dense: Camadas totalmente conectadas para predição final
    
    Args:
        seq_length: Comprimento da sequência de entrada
        regularization: Fator de regularização L2
    
    Returns:
        Modelo CNN compilado
    """
    model = Sequential([
        Input(shape=(seq_length, 1)),
        
        # Primeira camada convolucional
        Conv1D(filters=128, 
               kernel_size=2, 
               activation='relu',
               kernel_regularizer=regularizers.l2(regularization)),
        BatchNormalization(),
        
        # Segunda camada convolucional
        Conv1D(filters=64,
               kernel_size=2,
               activation='relu',
               kernel_regularizer=regularizers.l2(regularization)),
        
        # Max pooling para reduzir dimensionalidade
        MaxPooling1D(pool_size=2),
        
        # Flatten para preparar para camadas densas
        Flatten(),
        
        # Camadas totalmente conectadas
        Dense(50, activation='relu',
              kernel_regularizer=regularizers.l2(regularization)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(20, activation='relu',
              kernel_regularizer=regularizers.l2(regularization)),
        Dropout(0.2),
        
        # Camada de saída
        Dense(1, activation='linear')
    ], name='CNN_Model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse', 'mae']
    )
    
    logger.info(f"CNN criado: {model.count_params()} parâmetros")
    return model

def criar_callbacks():
    callbacks = [
        # Para parar o treinamento se não houver melhora
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduz learning rate quando não há progresso
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks

def comparar_arquiteturas():
    """
    Compara as características dos diferentes modelos.
    
    Returns:
        Dicionário com comparação dos modelos
    """
    comparacao = {
        'MLP': {
            'tipo': 'Feedforward',
            'memoria_temporal': 'Nenhuma (trata como vetor)',
            'melhor_para': 'Relações não-lineares simples',
            'complexidade': 'Baixa',
            'vantagens': ['Simples', 'Rápido', 'Pouco overfitting'],
            'desvantagens': ['Não captura dependências temporais']
        },
        
        'LSTM': {
            'tipo': 'Recorrente',
            'memoria_temporal': 'Curto e longo prazo (h(t) e c(t))',
            'melhor_para': 'Séries com dependências temporais complexas',
            'complexidade': 'Alta',
            'vantagens': ['Captura dependências longas', 'Lida com vanishing gradient'],
            'desvantagens': ['Computacionalmente custoso', 'Propenso a overfitting']
        },
        
        'CNN': {
            'tipo': 'Convolucional',
            'memoria_temporal': 'Local (tamanho do kernel)',
            'melhor_para': 'Padrões repetitivos e sazonalidade',
            'complexidade': 'Média',
            'vantagens': ['Detecta padrões locais', 'Parallelizável'],
            'desvantagens': ['Limitado a padrões locais', 'Requer muitos dados']
        }
    }
    
    return comparacao