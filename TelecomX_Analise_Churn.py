import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
with open("TelecomX_Data.json", "r") as file:
    data = json.load(file)

df = pd.json_normalize(data)

# Exploração inicial
print("Colunas e tipos de dados:")
print(df.info())
print("\nPrimeiras linhas:")
print(df.head())

# Verificação de inconsistências
print("\nValores ausentes por coluna:")
print(df.isnull().sum())
print("\nDuplicatas:", df.duplicated().sum())

# Limpeza dos dados
df = df[df['Churn'].isin(['Yes', 'No'])]
df['account.Charges.Total'] = pd.to_numeric(df['account.Charges.Total'], errors='coerce')
df = df.drop_duplicates()
df = df.dropna(subset=['account.Charges.Total', 'account.Charges.Monthly'])

# Nova coluna
df['Contas_Diarias'] = df['account.Charges.Monthly'] / 30

# Padronização
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['customer.Partner'] = df['customer.Partner'].map({'Yes': 1, 'No': 0})
df['customer.Dependents'] = df['customer.Dependents'].map({'Yes': 1, 'No': 0})
df['account.PaperlessBilling'] = df['account.PaperlessBilling'].map({'Yes': 1, 'No': 0})

# Análise descritiva
print("\nEstatísticas descritivas:")
print(df.describe())

# Visualização: Distribuição de Evasão
sns.countplot(x="Churn", data=df)
plt.title("Distribuição de Evasão")
plt.savefig("evasao.png")
plt.clf()

# Visualização: Churn por variáveis categóricas
cat_vars = ['customer.Gender', 'account.Contract', 'account.PaymentMethod']
for var in cat_vars:
    if var in df.columns:
        sns.countplot(x=var, hue="Churn", data=df)
        plt.title(f"Evasão por {var}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"evasao_{var}.png")
        plt.clf()

# Visualização: Churn por variáveis numéricas
num_vars = ['account.Charges.Total', 'account.Charges.Monthly', 'Contas_Diarias']
for var in num_vars:
    if var in df.columns:
        sns.boxplot(x="Churn", y=var, data=df)
        plt.title(f"{var} por Evasão")
        plt.tight_layout()
        plt.savefig(f"boxplot_{var}.png")
        plt.clf()

# Scatterplot entre Contas_Diarias e Churn
sns.scatterplot(x='Contas_Diarias', y='Churn', data=df, alpha=0.3)
plt.title("Contas Diárias vs Evasão")
plt.savefig("scatter_ContasDiarias_Churn.png")
plt.clf()

# Ajuste os nomes das colunas conforme o dicionário de dados
servicos = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
servicos_existentes = [col for col in servicos if col in df.columns]
if servicos_existentes:
    df['Qtd_Servicos'] = df[servicos_existentes].apply(lambda x: sum(x == 'Yes'), axis=1)
    num_vars.append('Qtd_Servicos')

# Correlação
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlação entre Variáveis")
plt.savefig("correlacao.png")
plt.clf()

# Visualização: Churn por Qtd_Servicos
if 'Qtd_Servicos' in df.columns:
    sns.boxplot(x="Churn", y="Qtd_Servicos", data=df)
    plt.title("Qtd_Servicos por Evasão")
    plt.tight_layout()
    plt.savefig("boxplot_Qtd_Servicos.png")
    plt.clf()
    print("\nCorrelação entre Qtd_Servicos e Churn:", df['Qtd_Servicos'].corr(df['Churn']))

print("Análise concluída. Gráficos salvos como imagens.")
