# 📊 Análise de Evasão de Clientes - Telecom X

Este projeto realiza uma análise exploratória de dados (EDA) para identificar padrões e fatores relacionados à evasão de clientes (churn) em uma empresa de telecomunicações fictícia, a Telecom X.

## 🔧 Tecnologias Utilizadas
- Python 3
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

## 📁 Estrutura
- `TelecomX_Analise_Churn.ipynb`: Notebook com toda a análise e visualizações.
- `TelecomX_Analise_Churn.py`: Script para análise exploratória e visualização de dados.
- `TelecomX_Analise_Churn.pdf`: Versão em PDF com relatório.
- `TelecomX_Data.json`: Base de dados fornecida.
- `TelecomX_Preprocessamento.py`: Script para pré-processamento dos dados, incluindo limpeza, encoding, balanceamento de classes com SMOTE e normalização. Prepara os dados para a modelagem de machine learning.
- `TelecomX_Modelagem.py`: Script que treina, avalia e compara os modelos de Regressão Logística e Random Forest. Também gera análises de importância de variáveis.
- `RELATORIO.md`: Relatório com as conclusões da análise.
- Imagens (`.png`): Gráficos gerados pela análise exploratória.

## 🚀 Como Executar
1. Clone o repositório.
2. Instale os pacotes necessários:
   ```bash
   pip install pandas matplotlib seaborn notebook
   ```
3. Execute o notebook:
   ```bash
   jupyter notebook TelecomX_Analise_Churn.ipynb
   ```
4. **Análise Exploratória**:
   ```bash
   python TelecomX_Analise_Churn.py
   ```
5. **Pré-processamento de Dados**:
   ```bash
   python TelecomX_Preprocessamento.py
   ```
6. **Treinamento e Avaliação de Modelos**:
   ```bash
   python TelecomX_Modelagem.py
   ```

Este comando irá executar todo o pipeline, desde a carga dos dados até a avaliação final dos modelos e a geração dos gráficos de importância de variáveis.

## 🧠 O que foi feito
- Importação e limpeza dos dados
- Criação de coluna auxiliar `Contas_Diarias`
- Padronização de dados
- Visualização de evasão
- Análise por variáveis categóricas e numéricas
- Correlação entre variáveis

## 📌 Conclusões
- Clientes com contrato mensal têm maior taxa de churn.
- Pagamento via "Electronic check" é comum entre clientes que evadem.

## 📢 Recomendações
- Incentivar contratos de longo prazo.
- Revisar o processo de cobrança via "Electronic check".

---

© 2025 - Projeto fictício para fins educacionais.
