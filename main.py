# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import logging
from typing import Dict, List, Tuple

# Imports dos módulos melhorados
from functions import (
    carregar_dados, add_dia, processar_snapshot, 
    criar_sequencia, validar_series_temporal
)
from models import (
    criar_mlp, criar_lstm, criar_cnn, 
    criar_callbacks, comparar_arquiteturas
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

class PreditorMetricasRede:
    """
    Classe principal para predição de métricas de redes temporais.
    """
    
    def __init__(self, script_dir: str):
        self.script_dir = script_dir
        self.scaler = StandardScaler()
        self.metricas_df = None
        self.modelos = {}
        self.resultados = {}
        
    def carregar_e_processar_dados(self, data_filename: str) -> pd.DataFrame:
        """
        Carrega e processa dados de comunicação.
        
        Args:
            data_filename: Nome do arquivo de dados
            
        Returns:
            DataFrame com métricas calculadas por dia
        """
        logger.info(f"=== CARREGANDO DADOS: {data_filename} ===")
        
        filepath = os.path.join(self.script_dir, 'data', data_filename)
        
        # Carregar dados
        df = carregar_dados(filepath, data_filename, self.script_dir)
        df = add_dia(df)
        
        # Processar snapshots diários
        logger.info("Processando snapshots diários...")
        metrics = []
        
        for day, group in df.groupby('day'):
            resultado = processar_snapshot(day, group)
            metrics.append(resultado)
            
            if day % 50 == 0:  # Log a cada 50 dias
                logger.info(f"Processado dia {day}")
        
        self.metricas_df = pd.DataFrame(metrics)
        
        if self.metricas_df.empty:
            raise ValueError("Nenhuma métrica foi calculada!")
        
        logger.info(f"Processamento concluído: {len(self.metricas_df)} dias")
        logger.info(f"Métricas disponíveis: {list(self.metricas_df.columns)}")
        
        return self.metricas_df
    
    def preparar_dados_treinamento(self, metrica: str = 'avg_degree', 
                                 seq_length: int = 14) -> Dict:
        """
        Prepara dados para treinamento dos modelos.
        
        Args:
            metrica: Nome da métrica a ser predita
            seq_length: Tamanho da janela temporal
            
        Returns:
            Dicionário com dados de treino/teste
        """
        logger.info(f"=== PREPARANDO DADOS PARA PREDIÇÃO DE {metrica.upper()} ===")
        
        if metrica not in self.metricas_df.columns:
            raise ValueError(f"Métrica '{metrica}' não encontrada!")
        
        # Extrair série temporal
        series = self.metricas_df[metrica].values
        
        # Validar série
        if not validar_series_temporal(series, metrica):
            raise ValueError(f"Série temporal de {metrica} é inválida!")
        
        # Normalizar dados
        series_normalized = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
        
        # Criar sequências
        X, y = criar_sequencia(series_normalized, seq_length)
        
        if X.shape[0] == 0:
            raise ValueError(f"Não foi possível criar sequências com seq_length={seq_length}")
        
        # Dividir treino/teste (80/20, sem embaralhar para manter ordem temporal)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Preparar dados para LSTM/CNN (precisam de dimensão extra)
        X_train_lstm = X_train[..., np.newaxis]
        X_test_lstm = X_test[..., np.newaxis]
        
        dados = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_lstm': X_train_lstm,
            'X_test_lstm': X_test_lstm,
            'seq_length': seq_length,
            'series_original': series,
            'metrica': metrica
        }
        
        logger.info(f"Dados preparados:")
        logger.info(f"  - Treino: {X_train.shape[0]} amostras")
        logger.info(f"  - Teste: {X_test.shape[0]} amostras")
        logger.info(f"  - Janela temporal: {seq_length} dias")
        
        return dados
    
    def treinar_modelos(self, dados: Dict) -> Dict:
        """
        Treina todos os modelos de rede neural.
        
        Args:
            dados: Dicionário com dados preparados
            
        Returns:
            Dicionário com modelos treinados
        """
        logger.info("=== TREINAMENTO DOS MODELOS ===")
        
        seq_length = dados['seq_length']
        callbacks = criar_callbacks()
        
        modelos = {}
        
        # ----- MODELO BASE -----
        logger.info("Modelo Base: Repetir valor anterior")
        # Não precisa treinamento, apenas lógica de predição
        
        # ----- MLP -----
        logger.info("Treinando MLP...")
        tf.keras.backend.clear_session()
        
        mlp = criar_mlp(seq_length)
        mlp.fit(
            dados['X_train'], dados['y_train'],
            validation_data=(dados['X_test'], dados['y_test']),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        modelos['MLP'] = mlp
        
        # ----- LSTM -----
        logger.info("Treinando LSTM...")
        tf.keras.backend.clear_session()
        
        lstm = criar_lstm(seq_length)
        lstm.fit(
            dados['X_train_lstm'], dados['y_train'],
            validation_data=(dados['X_test_lstm'], dados['y_test']),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        modelos['LSTM'] = lstm
        
        # ----- CNN -----
        logger.info("Treinando CNN...")
        tf.keras.backend.clear_session()
        
        cnn = criar_cnn(seq_length)
        cnn.fit(
            dados['X_train_lstm'], dados['y_train'],
            validation_data=(dados['X_test_lstm'], dados['y_test']),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        modelos['CNN'] = cnn
        
        self.modelos = modelos
        logger.info("Treinamento concluído!")
        
        return modelos
    
    def avaliar_modelos(self, dados: Dict) -> Dict:
        """
        Avalia todos os modelos e calcula métricas.
        
        Args:
            dados: Dicionário com dados de teste
            
        Returns:
            Dicionário com resultados da avaliação
        """
        logger.info("=== AVALIAÇÃO DOS MODELOS ===")
        
        resultados = {}
        
        # Desnormalizar dados de teste para cálculo correto das métricas
        y_test_original = self.scaler.inverse_transform(
            dados['y_test'].reshape(-1, 1)
        ).flatten()
        
        # ----- MODELO BASE -----
        # Previsão: repetir o valor anterior
        y_pred_base_norm = dados['y_test'][:-1]  # Excluir último valor
        y_true_base_norm = dados['y_test'][1:]   # Excluir primeiro valor
        
        y_pred_base = self.scaler.inverse_transform(
            y_pred_base_norm.reshape(-1, 1)
        ).flatten()
        y_true_base = self.scaler.inverse_transform(
            y_true_base_norm.reshape(-1, 1)
        ).flatten()
        
        mae_base = mean_absolute_error(y_true_base, y_pred_base)
        rmse_base = np.sqrt(mean_squared_error(y_true_base, y_pred_base))
        
        resultados['BASE'] = {
            'MAE': mae_base,
            'RMSE': rmse_base,
            'predictions': y_pred_base,
            'actual': y_true_base
        }
        
        # ----- MODELOS DE REDE NEURAL -----
        for nome, modelo in self.modelos.items():
            logger.info(f"Avaliando {nome}...")
            
            # Fazer predições
            if nome in ['LSTM', 'CNN']:
                y_pred_norm = modelo.predict(dados['X_test_lstm'], verbose=0)
            else:  # MLP
                y_pred_norm = modelo.predict(dados['X_test'], verbose=0)
            
            # Desnormalizar predições
            y_pred = self.scaler.inverse_transform(y_pred_norm).flatten()
            
            # Calcular métricas
            mae = mean_absolute_error(y_test_original, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
            
            resultados[nome] = {
                'MAE': mae,
                'RMSE': rmse,
                'predictions': y_pred,
                'actual': y_test_original
            }
            
            logger.info(f"{nome} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        self.resultados = resultados
        return resultados
    
    def gerar_relatorio(self, data_filename: str):
        """
        Gera relatório completo dos resultados.
        
        Args:
            data_filename: Nome do arquivo de dados para nomear os arquivos
        """
        logger.info("=== GERANDO RELATÓRIO ===")
        
        # Criar diretório para resultados
        results_dir = os.path.join(self.script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        data_base = os.path.splitext(data_filename)[0]
        
        # ----- TABELA DE RESULTADOS -----
        print("\n" + "="*60)
        print("RESULTADOS DA PREDIÇÃO")
        print("="*60)
        print(f"{'Modelo':<10} {'MAE':<10} {'RMSE':<10}")
        print("-"*30)
        
        for modelo, metricas in self.resultados.items():
            print(f"{modelo:<10} {metricas['MAE']:<10.4f} {metricas['RMSE']:<10.4f}")
        
        # ----- GRÁFICOS -----
        self._gerar_graficos(data_base, results_dir)
        
        # ----- SALVAR RESULTADOS -----
        self._salvar_resultados_csv(data_base, results_dir)
        
        logger.info(f"Relatório salvo em: {results_dir}")
    
    def _gerar_graficos(self, data_base: str, results_dir: str):
        """
        Gera gráficos de comparação dos modelos.
        """
        modelos = list(self.resultados.keys())
        maes = [self.resultados[m]['MAE'] for m in modelos]
        rmses = [self.resultados[m]['RMSE'] for m in modelos]
        
        # Gráfico MAE
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.bar(modelos, maes, color=['gray', 'blue', 'green', 'red'])
        plt.title('Comparação MAE')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Gráfico RMSE
        plt.subplot(1, 2, 2)
        plt.bar(modelos, rmses, color=['gray', 'blue', 'green', 'red'])
        plt.title('Comparação RMSE')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        grafico_path = os.path.join(results_dir, f'comparacao_modelos_{data_base}.png')
        plt.savefig(grafico_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de predições vs valores reais
        plt.figure(figsize=(15, 10))
        
        for i, (modelo, dados) in enumerate(self.resultados.items(), 1):
            plt.subplot(2, 2, i)
            
            actual = dados['actual']
            predictions = dados['predictions']
            
            # Ajustar tamanhos se necessário (modelo BASE tem menos pontos)
            if len(predictions) < len(actual):
                actual = actual[:len(predictions)]
            
            x = range(len(actual))
            plt.plot(x, actual, label='Real', color='black', linewidth=2)
            plt.plot(x, predictions, label='Predito', color='red', linewidth=1, alpha=0.7)
            plt.title(f'{modelo} - Predições vs Real')
            plt.xlabel('Tempo')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        predicoes_path = os.path.join(results_dir, f'predicoes_{data_base}.png')
        plt.savefig(predicoes_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráficos salvos: {grafico_path}, {predicoes_path}")
    
    def _salvar_resultados_csv(self, data_base: str, results_dir: str):
        """
        Salva resultados em arquivo CSV.
        """
        # Criar DataFrame com resultados
        resultados_df = pd.DataFrame([
            {'Modelo': modelo, 'MAE': dados['MAE'], 'RMSE': dados['RMSE']}
            for modelo, dados in self.resultados.items()
        ])
        
        csv_path = os.path.join(results_dir, f'resultados_{data_base}.csv')
        resultados_df.to_csv(csv_path, index=False)
        
        # Salvar métricas detalhadas
        metricas_path = os.path.join(results_dir, f'metricas_detalhadas_{data_base}.csv')
        self.metricas_df.to_csv(metricas_path, index=False)
        
        logger.info(f"Resultados CSV salvos: {csv_path}")

def main():
    """
    Função principal do programa.
    """
    # Configuração inicial
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Arquivos de dados disponíveis
    arquivos_dados = [
        'email-Eu-core-temporal.txt',
        'CollegeMsg.txt',
    ]
    
    # Métricas disponíveis para predição
    metricas_disponiveis = ['avg_degree', 'clustering', 'density']
    
    # Parâmetros de configuração
    seq_length = 14  # Janela temporal (conforme artigo)
    metrica_alvo = 'avg_degree'  # Métrica para predição
    
    # Selecionar arquivo de dados
    data_filename = arquivos_dados[0]  # Altere o índice para testar outros arquivos
    
    try:
        logger.info("=== INICIANDO PREDIÇÃO DE MÉTRICAS DE REDE ===")
        logger.info(f"Arquivo: {data_filename}")
        logger.info(f"Métrica: {metrica_alvo}")
        logger.info(f"Janela temporal: {seq_length} dias")
        
        # Criar preditor
        preditor = PreditorMetricasRede(script_dir)
        
        # Executar pipeline completo
        metricas_df = preditor.carregar_e_processar_dados(data_filename)
        dados = preditor.preparar_dados_treinamento(metrica_alvo, seq_length)
        modelos = preditor.treinar_modelos(dados)
        resultados = preditor.avaliar_modelos(dados)
        preditor.gerar_relatorio(data_filename)
        
        # Mostrar comparação de arquiteturas
        logger.info("\n=== COMPARAÇÃO DE ARQUITETURAS ===")
        comparacao = comparar_arquiteturas()
        for modelo, info in comparacao.items():
            print(f"\n{modelo}:")
            print(f"  Tipo: {info['tipo']}")
            print(f"  Memória temporal: {info['memoria_temporal']}")
            print(f"  Melhor para: {info['melhor_para']}")
            print(f"  Vantagens: {', '.join(info['vantagens'])}")
        
        logger.info("=== EXECUÇÃO CONCLUÍDA COM SUCESSO ===")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

if __name__ == "__main__":
    main()