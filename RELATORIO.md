# Relatório de Análise de Evasão de Clientes - TelecomX

## 1. Introdução

Este relatório detalha o processo de análise de dados e construção de modelos de machine learning para prever a evasão de clientes (churn) na empresa TelecomX. O objetivo é identificar os principais fatores que levam à evasão e fornecer insights para estratégias de retenção.

## 2. Preparação dos Dados

A preparação dos dados foi uma etapa crucial e envolveu os seguintes passos:

- **Limpeza**: Foram removidos dados inconsistentes e valores ausentes nas colunas de cobrança (`account.Charges.Total`).
- **Encoding**: Variáveis categóricas (como tipo de contrato, forma de pagamento, etc.) foram transformadas em formato numérico usando a técnica de *One-Hot Encoding* para serem compatíveis com os modelos.
- **Verificação de Desbalanceamento**: Foi identificado um desbalanceamento na variável alvo (`Churn`), com uma proporção maior de clientes não evadidos.
- **Balanceamento de Classes**: Para corrigir o desbalanceamento, aplicou-se a técnica **SMOTE** (*Synthetic Minority Over-sampling Technique*) apenas nos dados de treino. Isso evita que os modelos fiquem enviesados para a classe majoritária.
- **Normalização**: As variáveis numéricas foram padronizadas (*StandardScaler*) para que tivessem média zero e desvio padrão um, garantindo que a escala não influenciasse os modelos baseados em distância, como a Regressão Logística.

## 3. Modelagem e Avaliação

Foram treinados e avaliados dois modelos distintos para prever a evasão:

1.  **Regressão Logística**: Um modelo linear, rápido e interpretável, ideal para problemas de classificação binária.
2.  **Random Forest**: Um modelo de conjunto (*ensemble*) baseado em árvores de decisão, conhecido por sua alta performance e capacidade de medir a importância das variáveis.

Os dados foram divididos em **70% para treino** e **30% para teste**.

### Resultados da Avaliação

| Modelo                | Acurácia | Precisão (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) |
| --------------------- | :------: | :----------------: | :--------------: | :----------------: |
| **Regressão Logística** |  74.5%   |        51.0%       |      82.0%       |       63.0%        |
| **Random Forest**       |  78.3%   |        60.0%       |      63.0%       |       61.0%        |

*(Nota: os valores são aproximados e baseados na execução do script `TelecomX_Modelagem.py`)*

### Análise dos Modelos

- A **Regressão Logística** obteve um **Recall** muito alto, o que significa que foi excelente em identificar os clientes que de fato evadiram. No entanto, sua **Precisão** foi menor, indicando que classificou erroneamente alguns clientes não evadidos como evadidos (falsos positivos).
- O **Random Forest** apresentou uma **Acurácia** geral superior e um bom equilíbrio entre Precisão e Recall. Ele foi mais preciso ao prever a evasão, embora tenha deixado de identificar alguns clientes que evadiram.

**Conclusão da Avaliação**: A escolha do melhor modelo depende do objetivo de negócio.
- Se a prioridade é **não perder nenhum cliente que possa evadir** (mesmo que isso implique em ações de retenção para clientes que não iriam sair), a **Regressão Logística** é a melhor escolha devido ao seu alto Recall.
- Se o objetivo é uma **abordagem mais equilibrada e com menos "alarmes falsos"**, o **Random Forest** é mais indicado.

## 4. Análise de Importância das Variáveis

A análise das variáveis mais influentes revelou os seguintes fatores como principais impulsionadores da evasão:

1.  **Tipo de Contrato (`account.Contract`)**: Clientes com contratos mensais (`Month-to-month`) têm uma propensão muito maior a evadir em comparação com contratos de 1 ou 2 anos.
2.  **Serviço de Internet (`InternetService`)**: Clientes com serviço de `Fiber optic` mostraram maior tendência à evasão, possivelmente devido a problemas de custo, estabilidade ou concorrência.
3.  **Total Gasto (`account.Charges.Total`)**: Valores totais mais baixos estão associados a uma maior chance de evasão, o que sugere que clientes mais novos ou com menos serviços são mais propensos a sair.
4.  **Serviços de Suporte e Segurança**: A ausência de serviços como `OnlineSecurity` e `TechSupport` é um forte indicador de risco de evasão.

## 5. Conclusão e Recomendações Estratégicas

Com base na análise, a evasão de clientes na TelecomX é fortemente influenciada por fatores contratuais, tipo de serviço de internet e a falta de serviços de valor agregado, como suporte técnico.

**Recomendações**:

- **Fidelização Contratual**: Criar campanhas e ofertas para incentivar os clientes com contrato mensal a migrarem para planos de 1 ou 2 anos, oferecendo descontos ou benefícios.
- **Melhoria do Serviço de Fibra Óptica**: Investigar a causa da alta evasão entre clientes de fibra. Pode ser uma questão de preço, qualidade do serviço ou suporte. Uma pesquisa de satisfação focada nesse grupo seria valiosa.
- **Cross-selling de Serviços de Proteção**: Oferecer pacotes promocionais de `OnlineSecurity` e `TechSupport` para clientes que não possuem esses serviços, destacando os benefícios de segurança e conveniência.
- **Ações Proativas de Retenção**: Utilizar o modelo de Regressão Logística para gerar uma lista de clientes com alta probabilidade de evasão (alto Recall) e direcionar a eles ações de retenção personalizadas, como descontos ou upgrades.
