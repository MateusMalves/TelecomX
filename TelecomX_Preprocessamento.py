import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# --- 1. Carregamento e Limpeza Inicial ---
print("Carregando e limpando os dados...")
with open("TelecomX_Data.json", "r") as file:
    data = json.load(file)

df = pd.json_normalize(data)

# Remoção de Colunas Irrelevantes: o ID do cliente não agrega valor à predição.
if 'customer.customerID' in df.columns:
    print("Removendo a coluna de ID do cliente ('customer.customerID')...")
    df = df.drop(columns=['customer.customerID'])

# Limpeza de dados (similar ao script de análise)
df = df[df['Churn'].isin(['Yes', 'No'])]
df['account.Charges.Total'] = pd.to_numeric(df['account.Charges.Total'], errors='coerce')
df = df.dropna(subset=['account.Charges.Total'])
df = df.drop_duplicates().reset_index(drop=True)

# --- 2. Verificação da Proporção de Evasão ---
print("\nVerificando a proporção de evasão (Churn):")
churn_proportion = df['Churn'].value_counts(normalize=True)
print(churn_proportion)
if churn_proportion.min() < 0.4:
    print("Atenção: Classes desbalanceadas detectadas. Recomenda-se balanceamento.")

# --- 3. Separação de Features (X) e Alvo (y) ---
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# --- 4. Separação em Dados de Treino e Teste ---
print("\nSeparando dados em treino (70%) e teste (30%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Tamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho do teste: {X_test.shape[0]} amostras")

# --- 5. Pré-processamento com Pipelines ---
# Identificar tipos de colunas para pré-processamento
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Diagnóstico de cardinalidade
print("Cardinalidade das colunas categóricas:")
for col in categorical_features:
    print(f"{col}: {X_train[col].nunique()} valores únicos")

# Remover colunas categóricas com alta cardinalidade (ex: IDs disfarçados)
high_cardinality_cols = [col for col in categorical_features if X_train[col].nunique() > 50]
if high_cardinality_cols:
    print(f"Removendo colunas de alta cardinalidade: {high_cardinality_cols}")
    X_train = X_train.drop(columns=high_cardinality_cols)
    X_test = X_test.drop(columns=high_cardinality_cols)
    categorical_features = [col for col in categorical_features if col not in high_cardinality_cols]

# Pipeline para variáveis numéricas: Padronização
numeric_transformer = StandardScaler()

# Pipeline para variáveis categóricas: One-Hot Encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Criar o pré-processador com ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Mantém colunas não especificadas (se houver)
)

# Aplicar o pré-processamento (fit no treino, transform no treino e teste)
print("\nAplicando pré-processamento (padronização e one-hot encoding)...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- 6. Balanceamento de Classes com SMOTE (Apenas no Treino) ---
print("\nAplicando SMOTE para balancear as classes de treino...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print("\nProporção de classes após SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

# --- 7. Análise de Correlação (Pós-processamento) ---
# Para a correlação, criamos um DataFrame com os dados de treino reamostrados
# e os nomes das colunas pós OneHotEncoding
processed_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_processed_features = list(numeric_features) + list(processed_cat_features)

df_train_resampled = pd.DataFrame(X_train_resampled, columns=all_processed_features)
df_train_resampled['Churn'] = y_train_resampled

# Calcular e visualizar a correlação
print("\nGerando heatmap de correlação com os dados de treino processados...")
correlation_matrix = df_train_resampled.corr(numeric_only=True)
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.1f')
plt.title("Matriz de Correlação das Variáveis (Dados de Treino Pós-SMOTE)")
plt.savefig("correlacao_preprocessada.png")
plt.clf()
print("Heatmap salvo como 'correlacao_preprocessada.png'")


print("\n--- Pré-processamento concluído! ---")
print("Variáveis disponíveis para modelagem:")
print("X_train_resampled, y_train_resampled (dados de treino, processados e balanceados)")
print("X_test_processed, y_test (dados de teste, processados)") 