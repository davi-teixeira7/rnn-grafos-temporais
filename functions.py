# functions.py

import pandas as pd
import numpy as np
import networkx as nx
import os
from typing import Tuple, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def carregar_dados(filepath: str, data_filename: str, script_dir: str) -> pd.DataFrame:
    """
    Carrega dados de comunicação com validação robusta.
    
    Args:
        filepath: Caminho completo para o arquivo
        data_filename: Nome do arquivo
        script_dir: Diretório do script
    
    Returns:
        DataFrame com colunas SRC, TGT, TS
    """
    try:
        # Verificar se arquivo existe
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        # Tentar diferentes separadores
        separadores = [' ', '\t', ',']
        df = None
        
        for sep in separadores:
            try:
                df = pd.read_csv(filepath, sep=sep, header=None, names=['SRC', 'TGT', 'TS'])
                # Validar se temos 3 colunas
                if df.shape[1] == 3:
                    break
            except:
                continue
        
        if df is None or df.shape[1] != 3:
            raise ValueError("Não foi possível carregar o arquivo com formato correto")
        
        # Validações dos dados
        if df.empty:
            raise ValueError("Arquivo está vazio")
        
        # Verificar tipos de dados
        if not pd.api.types.is_numeric_dtype(df['TS']):
            logger.warning("Timestamp não é numérico, tentando converter...")
            df['TS'] = pd.to_numeric(df['TS'], errors='coerce')
        
        # Remover linhas com valores inválidos
        df = df.dropna()
        
        # Verificar se ainda temos dados
        if df.empty:
            raise ValueError("Nenhum dado válido após limpeza")
        
        logger.info(f"Arquivo carregado: {len(df)} registros")
        logger.info(f"Período: {df['TS'].min()} a {df['TS'].max()}")
        logger.info(f"Nós únicos: {len(set(df['SRC'].unique()).union(set(df['TGT'].unique())))}")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {filepath}")
        logger.error(f"Verifique se '{data_filename}' existe em '{script_dir}/data/'")
        raise
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

def add_dia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona coluna de dia baseada no timestamp.
    
    Args:
        df: DataFrame com coluna TS
    
    Returns:
        DataFrame com coluna 'day' adicionada
    """
    if 'TS' not in df.columns:
        raise ValueError("Coluna 'TS' não encontrada")
    
    # Converter timestamp para dias (assumindo segundos)
    df = df.copy()
    df['day'] = (df['TS'] // (24*60*60)).astype(int)
    
    # Normalizar dias para começar do 0
    df['day'] = df['day'] - df['day'].min()
    
    logger.info(f"Dados organizados em {df['day'].nunique()} dias")
    logger.info(f"Período: dia {df['day'].min()} a {df['day'].max()}")
    
    return df

def criar_sequencia(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências de janela deslizante para treinamento temporal.
    
    Args:
        data: Array de dados temporais
        seq_length: Tamanho da janela de entrada
    
    Returns:
        Tuple (X, y) onde X são as sequências de entrada e y os valores alvo
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if len(data) <= seq_length:
        logger.warning(f"Dados insuficientes: {len(data)} pontos para janela de {seq_length}")
        return np.array([]), np.array([])
    
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    logger.info(f"Criadas {len(X)} sequências de treinamento")
    
    return X, y

def processar_snapshot(day: int, group: pd.DataFrame) -> Dict[str, Any]:
    """
    Processa um snapshot diário e calcula métricas de rede.
    
    Args:
        day: Número do dia
        group: DataFrame com transações do dia
    
    Returns:
        Dicionário com métricas calculadas
    """
    try:
        # Criar grafo não-direcionado
        G = nx.Graph()
        
        # Adicionar arestas com peso (contagem de interações)
        edge_counts = {}
        for _, row in group.iterrows():
            src, tgt = row['SRC'], row['TGT']
            edge = tuple(sorted([src, tgt]))  # Garantir ordem consistente
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        # Adicionar arestas com pesos
        for (src, tgt), weight in edge_counts.items():
            G.add_edge(src, tgt, weight=weight)
        
        # Calcular métricas
        if len(G.nodes) > 0:
            # Grau médio
            degrees = [d for n, d in G.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            
            # Coeficiente de clustering médio
            clustering = nx.average_clustering(G)
            
            # Métricas adicionais
            density = nx.density(G)
            num_nodes = len(G.nodes)
            num_edges = len(G.edges)
            
            # Componentes conectados
            num_components = nx.number_connected_components(G)
            largest_cc_size = len(max(nx.connected_components(G), key=len)) if num_components > 0 else 0
            
        else:
            avg_degree = clustering = density = 0
            num_nodes = num_edges = num_components = largest_cc_size = 0
        
        return {
            'day': day,
            'avg_degree': avg_degree,
            'clustering': clustering,
            'density': density,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_components': num_components,
            'largest_cc_size': largest_cc_size,
            'total_transactions': len(group)
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar dia {day}: {e}")
        return {
            'day': day,
            'avg_degree': 0,
            'clustering': 0,
            'density': 0,
            'num_nodes': 0,
            'num_edges': 0,
            'num_components': 0,
            'largest_cc_size': 0,
            'total_transactions': 0
        }

def validar_series_temporal(series: np.ndarray, nome_metrica: str = "métrica") -> bool:
    """
    Valida se a série temporal é adequada para predição.
    
    Args:
        series: Array com valores da série temporal
        nome_metrica: Nome da métrica para logging
    
    Returns:
        True se a série é válida, False caso contrário
    """
    if len(series) < 10:
        logger.warning(f"{nome_metrica}: Série muito curta ({len(series)} pontos)")
        return False
    
    # Verificar se não é constante
    if np.std(series) == 0:
        logger.warning(f"{nome_metrica}: Série constante (sem variação)")
        return False
    
    # Verificar valores inválidos
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        logger.warning(f"{nome_metrica}: Contém valores inválidos (NaN/Inf)")
        return False
    
    logger.info(f"{nome_metrica}: Série válida - {len(series)} pontos, "
                f"média={np.mean(series):.3f}, std={np.std(series):.3f}")
    
    return True