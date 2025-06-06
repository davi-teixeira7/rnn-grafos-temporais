import matplotlib.pyplot as plt
import numpy as np
import os

def plotar_mae_individual(nomes_modelos, medias, desvios, titulo, nome_arquivo, diretorio='images'):
    plt.figure(figsize=(4, 4))
    x = np.arange(len(nomes_modelos))
    plt.errorbar(x, medias, yerr=desvios, fmt='o', capsize=5, linestyle='none', color='blue')
    plt.xticks(x, nomes_modelos)
    plt.ylabel('MAE')
    plt.title(titulo)
    plt.grid(True)
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, nome_arquivo)
    plt.savefig(caminho)
    plt.close()
    print(f"Gráfico salvo: {caminho}")
