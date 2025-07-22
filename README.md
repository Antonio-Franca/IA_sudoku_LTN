# Trabalho Final de IA: Sudoku com Redes Lógicas Tensoriais (LTN)

**Disciplina:** INTELIGENCIA ARTIFICAL 2025/01

**Aluno:** Antonio Mileysson França Bragança

**Matrícula:** 21850963

**Curso:** Engenharia da Computação

-----

Este repositório contém um notebook que explora a aplicação de **Redes Lógicas Tensoriais (LTN)** para resolver e analisar quebra-cabeças de Sudoku 4x4. O projeto demonstra uma progressão de tarefas com complexidade crescente, desde a validação de soluções completas até a recomendação de estratégias de resolução, integrando aprendizado de máquina sub-simbólico (PyTorch) com raciocínio lógico formal.


###  Visão Geral do Projeto

O objetivo central deste trabalho é demonstrar como o paradigma de **IA Neuro-Simbólica**, através do framework LTNtorch, pode ser aplicado a problemas que exigem raciocínio lógico e estruturado, como o Sudoku. Em vez de uma abordagem de força bruta ou de aprendizado profundo convencional (black-box), utilizamos LTN para injetar o conhecimento de domínio (as regras do Sudoku) diretamente na arquitetura do modelo.

O notebook está estruturado em três fases principais:

1.  **Classificação de Satisfatibilidade**: Treinar um modelo para determinar se um tabuleiro 4x4 preenchido é uma solução válida.
2.  **Análise de Potencial**: Utilizar o primeiro modelo de forma híbrida com heurísticas para avaliar e classificar os melhores movimentos possíveis em um tabuleiro incompleto.
3.  **Recomendação de Estratégias**: Treinar um modelo mais avançado que aprende a reconhecer padrões e a recomendar a heurística de resolução mais apropriada (ex: *Naked Single*, *Hidden Single*) para o estado atual do jogo.

###  Principais Funcionalidades

  - **Validação Lógica**: Um modelo que aprende as regras do Sudoku e classifica tabuleiros completos como "Válidos" ou "Inválidos" com alta acurácia.
  - **Análise Híbrida de Movimentos**: Combinação de heurísticas clássicas com o modelo LTN para pontuar e ranquear os movimentos mais promissores em um tabuleiro aberto.
  - **Recomendação de Heurísticas**: Uma arquitetura neuro-simbólica avançada que classifica o estado de um tabuleiro e recomenda a estratégia de resolução mais eficiente a ser aplicada.
  
###  Tecnologias Utilizadas

  - **Python 3**
  - **PyTorch**: Framework base para tensores e treinamento de redes neurais.
  - **LTNtorch**: Biblioteca para integração de lógica de primeira ordem com PyTorch.
  - **NumPy**: Manipulação de arrays para estruturas de dados do tabuleiro.
  - **Pandas**: Leitura e escrita de arquivos CSV para os dados de teste.
  - **Scikit-learn**: Divisão de datasets em conjuntos de treino e validação.

### Estrutura do Projeto

O notebook está organizado de forma sequencial e modular:

1.  **Instalação e Configuração**: Setup do ambiente e importação das bibliotecas.
2.  **Geração de Dados**: Funções para criar programaticamente datasets de tabuleiros válidos, inválidos e incompletos.
3.  **Modelo 1: `SudokuLTN`**: Definição e treinamento do modelo classificador de tabuleiros completos.
4.  **Modelo 2: `LTNMoveAnalyzer`**: Definição e demonstração do sistema híbrido para análise de movimentos.
5.  **Modelo 3: `LTNHeuristicRecommender4x4`**: Definição e treinamento do modelo avançado para recomendação de estratégias.
6.  **Avaliação e Demonstração**: Células dedicadas ao final de cada seção para carregar e testar os modelos treinados em exemplos práticos.

###  Arquitetura dos Modelos

#### Modelo 1: `SudokuLTN` - Validador de Tabuleiros Completos

  - **Objetivo**: Classificação binária (solução válida/inválida).
  - **Arquitetura**: Utiliza predicados LTN que correspondem às três regras fundamentais do Sudoku: unicidade de valores em linhas, colunas e blocos. A fórmula lógica `∀r∈rows, P_row(r) ∧ ∀c∈cols, P_col(c) ∧ ∀b∈blocks, P_block(b)` é implementada diretamente no `forward pass`. O modelo aprende a definição de um conjunto "válido" e agrega os resultados para produzir um único valor de verdade para o tabuleiro inteiro.

#### Modelo 2: `LTNMoveAnalyzer` - Analisador Híbrido de Movimentos

  - **Objetivo**: Rankear os melhores movimentos possíveis em um tabuleiro incompleto.
  - **Arquitetura**: Híbrida.
    1.  **`SudokuHeuristicAnalyzer`**: Um componente baseado em regras que realiza verificações rápidas para identificar estados sem solução.
    2.  **`LTNClassifier`**: Uma interface que utiliza o **Modelo 1** treinado. Para cada movimento válido possível, ele preenche temporariamente uma célula vazia com um valor neutro e usa o modelo para obter uma "pontuação de potencial", avaliando o quão "próximo" de uma solução válida o novo estado parece.
  - **Resultado**: Uma lista de movimentos classificada da mais alta à mais baixa probabilidade de levar a uma solução correta.

#### Modelo 3: `LTNHeuristicRecommender4x4` - Recomendador de Heurísticas

  - **Objetivo**: Classificar o estado de um tabuleiro e recomendar a estratégia de resolução mais adequada.
  - **Arquitetura**: Neuro-Simbólica.
    1.  **Entrada**: Um tensor de features multi-canal $(3 \\times 4 \\times 4)$ representando os valores, a contagem de candidatos e os conflitos de cada célula.
    2.  **Predicados Especialistas**: O modelo possui predicados LTN que funcionam como extratores de características treináveis, cada um especializado em detectar o potencial de uma heurística específica (`NakedSinglePotential`, `HiddenSinglePotential`, etc.).
    3.  **Agregação Lógica**: O quantificador existencial (∃) agrega as evidências encontradas pelos predicados em todo o tabuleiro, gerando um vetor de características semânticas que representa a confiança em cada heurística.
    4.  **Camada de Decisão**: Uma camada linear final classifica este vetor semântico, produzindo uma recomendação sobre qual heurística deve ser aplicada.

###  Como Executar

1.  **Clone o Repositório ou Baixe o Notebook**:
    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    ```
2.  **Abra no Google Colab**:
      - Acesse [colab.research.google.com](https://colab.research.google.com).
      - Clique em `File > Upload notebook...` e selecione o arquivo `.ipynb` clonado.

3.  **Execute as Células**:
      - Carregue os arquivos .csv no google colab
      - Para garantir a reprodutibilidade, execute as células em ordem sequencial.



