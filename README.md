# 🎯 Projeto: Detecção Precoce de Autismo com Machine Learning

Este projeto visa desenvolver um sistema baseado em **aprendizado de máquina** para a detecção precoce do autismo por meio da análise de **imagens faciais**. O sistema utiliza **medições antropométricas** obtidas a partir de **landmarks faciais**, extraídos através de técnicas de processamento de imagem. A estrutura do projeto está organizada conforme descrito abaixo.

## 📂 Estrutura do Projeto

```bash
project_root/
│
├── data/                          # Diretório para armazenar dados
│   ├── raw/                       # Dados brutos (não processados)
│   │   ├── autism_dataset/        # Conjunto de dados sobre autismo
│   │   │   ├── no_autism/         # Dados de indivíduos sem autismo
│   │   │   ├── with_autism/       # Dados de indivíduos com autismo
│   │   │   └── ...                # Outros subdiretórios conforme necessário
│   │   └── processed/             # Dados processados (prontos para uso)
│   │       └── ...                # Estrutura dos dados processados
│   └── README.md                  # Documentação sobre os dados
│
├── notebooks/                     # Notebooks Jupyter para análise
│   └── exploratory_analysis.ipynb # Análise exploratória dos dados
│
├── src/                           # Código-fonte do projeto
│   ├── __init__.py                # Torna o diretório um pacote Python
│   ├── data_processing.py         # Processamento de dados brutos
│   ├── feature_extraction.py      # Extração de características
│   ├── model_training.py          # Treinamento do modelo
│   ├── model_evaluation.py        # Avaliação do modelo
│   └── utils.py                   # Funções auxiliares
│
├── tests/                         # Testes automatizados
│   ├── __init__.py                # Torna o diretório um pacote Python
│   ├── test_data_processing.py    # Testes de processamento de dados
│   ├── test_feature_extraction.py # Testes de extração de características
│   └── ...                        # Outros testes conforme necessário
│
├── docs/                          # Documentação do projeto
│   ├── index.md                   # Índice da documentação
│   └── usage.md                   # Como usar o projeto
│
├── .gitignore                     # Arquivos a serem ignorados pelo Git
├── README.md                      # Documentação principal do projeto
├── requirements.txt               # Dependências do projeto
└── setup.py                       # Configuração do pacote
```

## 🧠 Topologia do Modelo de Machine Learning

## ⚙️ Estrutura do Modelo

O modelo de detecção de autismo em imagens faciais é composto por três etapas principais:

### 🧩 Etapa 1: Extração de Landmark Facial

A primeira etapa utiliza uma **CNN pré-treinada** para detectar landmarks faciais. Serão exploradas três abordagens:

- **Haarcascade com OpenCV (cv2)**: Detecta landmarks faciais com **68 pontos de referência**.
- **dlib**: Detecta landmarks faciais com **68 pontos de referência**, usando o dlib.
- **MediaPipe (Google)**: Utiliza um modelo de **malha facial 3D** com **438 pontos de referência**.

🔹 **A saída será um conjunto de landmarks faciais**, com 68 pontos para Haarcascade e dlib, e 438 pontos para MediaPipe.

### 🧮 Etapa 2: Cálculo de Medições Antropométricas

Com os landmarks extraídos, são calculadas **8 medições antropométricas** usando a fórmula da **Distância Euclidiana**. Os resultados são armazenados em um arquivo CSV.

📊 **Medições**:

| Referência 1           | Referência 2           | Distância  | Definição                |
|------------------------|------------------------|------------|--------------------------|
| Trichion (tr)           | Glabella (gl)          | Vertical   | Altura facial superior    |
| Glabella (gl)           | Filtrum superior (pu)  | Vertical   | Altura facial média       |
| Parte superior (pu)     | Menton (me)            | Vertical   | Altura facial inferior    |
| Filtrum superior (pu)   | Filtrum inferior (pl)  | Vertical   | Filtrum                  |
| Endo canthus (enl)      | Endo canthus (enr)     | Horizontal | Largura intercantal       |
| Exo canthus (exl)       | Exo canthus (exr)      | Horizontal | Largura biocular          |
| Alare (all)             | Alare (alr)            | Horizontal | Largura nasal             |
| Cheilion (chr)          | Cheilion (chl)         | Horizontal | Largura da boca           |

### 📊 Etapa 3: Classificação com Redes Neurais e Algoritmos de Machine Learning

Os dados extraídos (CSV com medições antropométricas) serão utilizados para treinar um modelo de classificação. Serão experimentados três algoritmos:

- **CNN (Redes Neurais Convolutivas)**: Para classificação baseada em medições antropométricas.
- **K-Nearest Neighbors (KNN)**: Para comparação de precisão.
- **Random Forest**: Para comparação de desempenho e precisão.

📍 **O objetivo é prever se a criança apresenta sinais de autismo.**

## 📋 Padronização do Repositório e Ferramentas Utilizadas

### 🗂️ Estrutura e Configuração do Repositório

A estrutura do repositório segue um padrão para facilitar a organização e manutenção do projeto. É utilizado o sistema de controle de versão Git com o modelo de branches Git Flow.

🔧 **Organização do repositório** inclui as pastas descritas acima, com detalhes a seguir:

- **data/**: Armazena dados brutos e processados.
- **notebooks/**: Armazena Jupyter Notebooks para análise exploratória.
- **src/**: Contém o código-fonte do projeto, dividido em módulos.
- **tests/**: Armazena testes automatizados.
- **docs/**: Documentação do projeto.

### 🛠️ Controle de Versão e Colaboração

Utilize branches para funcionalidades (**feature/**), correções (**bugfix/**), e versões (**release/**), garantindo organização e rastreabilidade.

### 💻 Configuração do Ambiente de Desenvolvimento

O ambiente de desenvolvimento utiliza o **Visual Studio Code** com extensões para **Python**, **Git**, e **Docker**. Ferramentas adicionais incluem:

- **Pylint**: Para linting.
- **Black**: Para formatação automática de código.
- **Jupyter Notebooks**: Para experimentação e análise de dados.

### 🔧 Linguagem de Programação e Bibliotecas

O sistema é desenvolvido em **Python**, utilizando bibliotecas como:

- **TensorFlow** e **Keras**: Para desenvolvimento e treinamento do modelo.
- **OpenCV**: Para processamento de imagens.
- **Pandas** e **NumPy**: Para manipulação de dados.

### ✅ Controle de Qualidade do Código

Para garantir a qualidade do código, são utilizadas ferramentas como **Pylint** (linting) e **Black** (formatação automática). O projeto também utiliza **integração contínua (CI)** com **GitHub Actions**, assegurando que o código submetido passe por testes automatizados.

## 🛠️ Instruções de Uso

### 1. **Clonando o Repositório**

Primeiro, faça o clone deste repositório para sua máquina local:

```bash
git clone https://github.com/GeorgeAlexsander/AutismDetectWithML.git
cd AutismDetectWithML
```

### 2. **Criando e Ativando um Ambiente Virtual**

Para garantir que as dependências do projeto não entrem em conflito com outras bibliotecas instaladas globalmente em seu sistema, é recomendado o uso de um ambiente virtual.

### No Windows

- Crie o ambiente virtual:

```bash
py -m venv venv
```

### No macOS/Linux

- Ative o ambiente virtual:

```bash
source venv/bin/activate
```

---

### 3. Instalando Dependências

Com o ambiente virtual ativo, você pode instalar todas as dependências do projeto diretamente a partir do arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Esse comando garantirá que todas as bibliotecas necessárias para rodar o projeto e gerar a documentação estejam instaladas.

**Importante**: O Sphinx precisa conseguir acessar e executar o código que está na pasta do projeto. No projeto temos os códigos-fonte organizados em uma subpasta, como `src/`, então é necessário realizar a instalação das dependências também na pasta onde o código fonte se encontra. Isso é crucial porque o Sphinx utiliza a extensão `autodoc`, que permite a geração automática de documentação a partir das docstrings do código. Para que o `autodoc` funcione corretamente, ele precisa acessar os módulos e suas respectivas dependências, garantindo assim que a documentação gerada reflita com precisão o funcionamento do código.

---

### 4. Gerando a Documentação

A documentação do projeto é gerada utilizando o **Sphinx**. Certifique-se de que o ambiente virtual está ativo e que todas as dependências foram instaladas.

- Navegue até o diretório da documentação:

```bash
cd docs
```

- Para gerar a documentação em HTML, execute o seguinte comando:

```bash
make html
```

- Caso haja a criação de novos arquivos de código a serem documentados, execute o comando abaixo para gerar automaticamente a documentação dos novos módulos, execute na pasta docs o comando a seguir:

```bash
sphinx-apidoc -o . ../src
```

Sendo que o ultimo termo, é o Path para os códigos-fontes, no exemplo eles estão na pasta src.

A documentação será gerada no diretório `docs/_build/html`. Abra o arquivo `index.html` em um navegador para visualizar a documentação.
