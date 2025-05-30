
# Predição de Métricas em Grafos Temporais com RNN

Este repositório contém código para predição de métricas em grafos temporais utilizando modelos de redes neurais como MLP, LSTM e CNN, desenvolvido para a disciplina de Fundamentos de IA.

## ✅ Criando ambiente virtual Python

Recomenda-se o uso de Python **3.8** ou superior.

1. Crie o ambiente virtual:

```bash
python -m venv venv
```

2. Ative o ambiente virtual:

- **Windows:**

```bash
venv\Scripts\activate
```

- **Linux/MacOS:**

```bash
source venv/bin/activate
```

---

## ✅ Instalando dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

**Ou, se preferir, instale manualmente:**

```bash
pip install pandas numpy networkx matplotlib scikit-learn tensorflow
```

---

## ✅ Rodando o `main.py`

Execute o script principal com:

```bash
python main.py
```

Certifique-se de que:  
- Os arquivos de dados `.txt` estejam na pasta `data/`.  
- O script `main.py` esteja na raiz do repositório.  
- A pasta `images/` exista para armazenar os gráficos gerados automaticamente.

---
