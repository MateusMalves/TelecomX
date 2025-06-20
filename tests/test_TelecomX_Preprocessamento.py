import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import unittest
import numpy as np

class TestTelecomXPreprocessamento(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
        })

    def test_carregamento_limpeza_dados(self):
        self.assertEqual(len(self.data), 5)

    def test_verificacao_proporcao_evasao(self):
        churn_proporcao = self.data['Churn'].value_counts(normalize=True)
        self.assertAlmostEqual(churn_proporcao['Yes'], 0.4)
        self.assertAlmostEqual(churn_proporcao['No'], 0.6)

    def test_separacao_dados_treino_teste(self):
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y
        )
        self.assertEqual(len(X_train), 3)
        self.assertEqual(len(X_test), 2)

    def test_aplicacao_smote(self):
        X, y = make_classification(n_classes=2, class_sep=2,
                                   weights=[0.9, 0.1], n_informative=3,
                                   n_redundant=1, flip_y=0,
                                   n_features=20, n_clusters_per_class=1,
                                   n_samples=1000, random_state=10)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.assertEqual(len(y_resampled[y_resampled == 0]), len(y_resampled[y_resampled == 1]))

    def test_erro_shape_dados(self):
        with self.assertRaises(ValueError):
            pd.DataFrame([[1]] * 7228, columns=[f'feature{i}' for i in range(4967)])

    # --- NOVOS TESTES ---

    def test_preprocessamento_pipeline(self):
        X = self.data.drop('Churn', axis=1)
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        X_processed = preprocessor.fit_transform(X)
        self.assertEqual(X_processed.shape[0], X.shape[0])

    def test_onehotencoder_output_shape(self):
        df = pd.DataFrame({
            'num': [1, 2, 3],
            'cat': ['a', 'b', 'a']
        })
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['num']),
            ('cat', OneHotEncoder(), ['cat'])
        ])
        X_processed = preprocessor.fit_transform(df)
        # 1 numeric + 2 categories = 3 columns
        self.assertEqual(X_processed.shape[1], 3)

    def test_smote_balanced_classes(self):
        X = np.random.rand(100, 2)
        y = np.array([0]*90 + [1]*10)
        smote = SMOTE(random_state=0)
        X_res, y_res = smote.fit_resample(X, y)
        self.assertEqual(sum(y_res == 0), sum(y_res == 1))

    def test_drop_duplicates(self):
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': [2, 2, 3],
            'Churn': ['No', 'No', 'Yes']
        })
        df_clean = df.drop_duplicates()
        self.assertEqual(len(df_clean), 2)

    def test_dropna_total_charges(self):
        df = pd.DataFrame({
            'account.Charges.Total': [10, None, 20],
            'Churn': ['No', 'Yes', 'No']
        })
        df_clean = df.dropna(subset=['account.Charges.Total'])
        self.assertEqual(len(df_clean), 2)

if __name__ == '__main__':
    unittest.main()