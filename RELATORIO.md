
# 📄 RELATÓRIO FINAL: Análise de Churn da Telecom X

## 1. Introdução

O objetivo desta análise foi explorar um conjunto de dados de clientes da empresa fictícia **Telecom X** para entender os principais fatores que levam à evasão de clientes (churn). Compreender por que os clientes cancelam seus serviços é o primeiro passo para desenvolver estratégias eficazes de retenção, reduzir perdas de receita e melhorar a satisfação do cliente.

## 2. Limpeza e Tratamento de Dados

O processo de preparação dos dados foi crucial para garantir a qualidade da análise e envolveu as seguintes etapas:

- **Importação:** Os dados, originalmente em formato JSON aninhado, foram carregados e convertidos em um DataFrame tabular utilizando a biblioteca `pandas`.
- **Tratamento de Inconsistências:** A coluna de faturamento total (`TotalCharges`) apresentava valores em branco para clientes novos. Esses valores foram tratados e a coluna convertida para o formato numérico.
- **Engenharia de Features:** Criamos a coluna `Contas_Diarias` a partir da fatura mensal, dividindo por 30, permitindo uma análise mais granular do custo para o cliente.
- **Padronização:** Valores textuais como `"Yes"` e `"No"` foram convertidos para 1 e 0. As colunas foram renomeadas para o português para facilitar a interpretação.

## 3. Análise Exploratória e Insights

### 3.1. Perfil Geral da Evasão

A taxa de evasão geral foi de **26,5%**. Isso significa que **mais de um quarto** dos clientes cancelaram os serviços — um número expressivo que exige atenção.

### 3.2. Fatores Contratuais

- **Contrato Mês a Mês:** Apresenta a maior taxa de churn. A flexibilidade do plano facilita o cancelamento.
- **Contratos de 1 ou 2 anos:** Clientes mais fiéis e com taxas de churn muito baixas.

### 3.3. Fatores de Custo e Serviço

- **Fatura Mensal Alta:** Clientes com valores elevados, especialmente usuários de fibra óptica, evadem com mais frequência.
- **Suporte Técnico:** A ausência de suporte técnico está fortemente associada à evasão. Clientes sem suporte tendem a cancelar.

### 3.4. Perfil do Cliente

- **Tempo de Contrato (Tenure):** A evasão é mais comum entre clientes novos. A lealdade aumenta com o tempo.
- **Outros Fatores:** Variáveis como gênero ou presença de parceiro(a) não influenciaram significativamente a evasão.

## 4. Conclusões

O cliente com maior risco de churn apresenta o seguinte perfil:

- **Cliente recente**
- **Contrato mês a mês**
- **Fatura alta (principalmente fibra óptica)**
- **Sem suporte técnico ou serviços adicionais**

## 5. Recomendações Estratégicas

### ✅ Incentivar Contratos de Longo Prazo
Campanhas e promoções para migração de contratos mensais para anuais ou bienais, com descontos progressivos.

### ✅ Revisar a Oferta de Fibra Óptica
Verificar a percepção de valor, qualidade do serviço e concorrência para justificar o custo elevado.

### ✅ Fortalecer o Onboarding
Criar programas de boas-vindas com suporte proativo nos primeiros 3 a 6 meses.

### ✅ Agregar Serviços de Suporte
Oferecer suporte técnico e proteção de dispositivos como diferencial competitivo nos planos.
