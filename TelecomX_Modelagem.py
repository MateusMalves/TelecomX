import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. Carregamento e Limpeza (Reutilizado do script de pré-processamento) ---
print("Iniciando pipeline de modelagem...")
print("Carregando e limpando os dados...")
with open("TelecomX_Data.json", "r") as file:
    data = json.load(file)

df = pd.json_normalize(data)
df = df[df['Churn'].isin(['Yes', 'No'])]
df['account.Charges.Total'] = pd.to_numeric(df['account.Charges.Total'], errors='coerce')
df = df.dropna(subset=['account.Charges.Total'])
df = df.drop_duplicates().reset_index(drop=True)

# --- 2. Separação e Pré-processamento (Reutilizado) ---
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- 3. Pipeline Completo (Processamento + Balanceamento) ---
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# --- 4. Criação, Treinamento e Avaliação de Modelos ---
models = {
    "Regressão Logística": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    print(f"\n--- Treinando e Avaliando: {name} ---")
    
    # Treinamento
    model.fit(X_train_resampled, y_train_resampled)
    
    # Previsão
    y_pred = model.predict(X_test_processed)
    
    # Avaliação
    print("Métricas de Avaliação:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Evadiu', 'Evadiu'], 
                yticklabels=['Não Evadiu', 'Evadiu'])
    plt.title(f'Matriz de Confusão - {name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.savefig(f"matriz_confusao_{name.replace(' ', '_')}.png")
    plt.clf()
    print(f"Matriz de confusão salva como 'matriz_confusao_{name.replace(' ', '_')}.png'")

# --- 5. Análise de Importância das Variáveis ---
print("\n--- Análise de Importância das Variáveis ---")

# Obter nomes das features após o one-hot encoding
processed_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_processed_features = list(numeric_features) + list(processed_cat_features)

# Para Random Forest
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=all_processed_features)
top_importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_importances, y=top_importances.index)
plt.title('Top 10 Variáveis Mais Importantes - Random Forest')
plt.savefig("importancia_variaveis_rf.png")
plt.clf()
print("\nGráfico de importância de variáveis do Random Forest salvo.")

# Para Regressão Logística
lr_model = models['Regressão Logística']
coefficients = pd.Series(lr_model.coef_[0], index=all_processed_features)
top_coefficients = coefficients.abs().sort_values(ascending=False).head(10)
top_coefficients = coefficients[top_coefficients.index]

plt.figure(figsize=(10, 8))
sns.barplot(x=top_coefficients, y=top_coefficients.index)
plt.title('Top 10 Coeficientes de Maior Impacto - Regressão Logística')
plt.savefig("importancia_variaveis_lr.png")
plt.clf()
print("Gráfico de coeficientes da Regressão Logística salvo.")

print("\n--- Modelagem e Análise concluídas! ---") 