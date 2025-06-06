# Projeto Python com Machine Learning

Este projeto utiliza bibliotecas de machine learning e análise de dados como TensorFlow, scikit-learn, pandas e outras.

## Configuração do Ambiente

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

### Configuração do Ambiente Virtual

#### 1. Clone o repositório (se aplicável)

```bash
git clone <seu-repositorio>
cd <nome-do-projeto>
```

#### 2. Crie um ambiente virtual

```bash
# No Windows
python -m venv venv

# No macOS/Linux
python3 -m venv venv
```

#### 3. Ative o ambiente virtual

**Windows:**

```bash
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

#### 4. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Execute o programa principal

```bash
python main.py
```

### Desativar o ambiente virtual

Quando terminar de trabalhar no projeto, você pode desativar o ambiente virtual:

```bash
deactivate
```

## Dependências

O projeto utiliza as seguintes bibliotecas principais:

- **pandas**: Manipulação e análise de dados
- **numpy**: Computação numérica
- **matplotlib**: Visualização de dados
- **scikit-learn**: Machine learning
- **tensorflow**: Deep learning
- **networkx**: Análise de redes e grafos

## Estrutura do Projeto

```
projeto/
│
├── main.py              # Arquivo principal
├── requirements.txt     # Dependências do projeto
├── README.md           # Este arquivo
└── venv/               # Ambiente virtual (criado após setup)
```

## Solução de Problemas

### Erro de importação do TensorFlow

Se você encontrar erros com o TensorFlow, tente:

```bash
pip install tensorflow-cpu  # Para versão apenas CPU
```

### Problemas com o ambiente virtual no Windows

Se tiver problemas para ativar o ambiente virtual no Windows, execute:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Atualizar dependências

Para atualizar todas as dependências:

```bash
pip install --upgrade -r requirements.txt
```

## Notas Adicionais

- Certifique-se de que o ambiente virtual está ativo antes de executar o código
- O arquivo `main.py` deve estar no diretório raiz do projeto
- Se você estiver usando um IDE como VS Code, selecione o interpretador Python do ambiente virtual
