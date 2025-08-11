import pandas as pd
import xgboost as xgb
import re
import nltk
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from skmultilearn.model_selection import iterative_train_test_split
from scipy.sparse import vstack
from nltk.corpus import stopwords
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pickle
import os

nltk.download('stopwords')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HierarchicalLocalPerNodeClassifier:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
        self.models = {}
        self.predictions = {}

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        stop_words = set(stopwords.words('portuguese'))
        words = [word for word in text.split() if word not in stop_words]
        return ' '.join(words)

    def load_and_prepare_data(self, file_path):
        try:
            logging.info("Carregando os dados...")
            data = pd.read_csv(file_path)
            data['text'] = data['text'].apply(self.clean_text)
            X = data['text']
            y = data.drop(columns=['text'])
            return X, y
        except FileNotFoundError:
            logging.error(f"Arquivo {file_path} não encontrado.")
            return None, None

    def filter_labels(self, y, min_count=10):
        label_counts = y.sum(axis=0)
        valid_labels = label_counts[label_counts >= min_count].index
        return y[valid_labels]

    def train_binary_classifier(self, X_pos, X_neg, y_pos, y_neg):
        X_train = vstack([X_pos, X_neg])
        y_train = y_pos + y_neg
        model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        model.fit(X_train, y_train)
        return model

    def train_and_predict(self, X_train, y_train, X_test, y_test, label_columns):
        self.models = {}
        self.predictions = {}
        for label in label_columns:
            idx = label_columns.index(label)
            y_train_col = y_train[:, idx]
            y_test_col = y_test[:, idx]

            pos_train = int((y_train_col == 1).sum())
            neg_train = int((y_train_col == 0).sum())
            logging.info(f"\n🔷 [Nó] {label} - Treinando com {pos_train} positivos, {neg_train} negativos")

            if pos_train == 0 or neg_train == 0:
                logging.warning(f"⚠️ Ignorando '{label}' (dados insuficientes)")
                continue

            X_pos = X_train[y_train_col == 1]
            X_neg = X_train[y_train_col == 0]
            y_pos = [1] * X_pos.shape[0]
            y_neg = [0] * X_neg.shape[0]

            model = self.train_binary_classifier(X_pos, X_neg, y_pos, y_neg)
            self.models[label] = model

            y_pred = model.predict(X_test)
            self.predictions[label] = y_pred
            logging.info(f"\n📊 Relatório para '{label}':\n{classification_report(y_test_col, y_pred, zero_division=0)}")

        self.apply_hierarchical_correction(label_columns)

    def apply_hierarchical_correction(self, label_columns):
        logging.info("🧩 Aplicando correção hierárquica...")
        reverse_hierarchy = self.build_reverse_hierarchy()

        for child, parents in reverse_hierarchy.items():
            if child not in self.predictions:
                continue
            for i in range(len(self.predictions[child])):
                if self.predictions[child][i] == 1:
                    for parent in parents:
                        if parent in self.predictions and self.predictions[parent][i] == 0:
                            self.predictions[parent][i] = 1

    def build_reverse_hierarchy(self):
        reverse = {}

        def recurse(parent, children):
            if isinstance(children, list):
                for child in children:
                    reverse.setdefault(child, []).append(parent)
            elif isinstance(children, dict):
                for child, subchildren in children.items():
                    reverse.setdefault(child, []).append(parent)
                    recurse(child, subchildren)

        for parent, children in self.hierarchy.items():
            recurse(parent, children)
        return reverse

    def gerar_particoes_multilabel(self, X_tfidf, y, n_splits=10, caminho='particoes.pkl'):
        logging.info(f"📁 Gerando {n_splits} partições multilabel para validação cruzada...")

        os.makedirs(os.path.dirname(caminho), exist_ok=True)

        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = []

        for train_idx, test_idx in mskf.split(X_tfidf, y):
            folds.append((train_idx, test_idx))

        with open(caminho, 'wb') as f:
            pickle.dump(folds, f)

        logging.info(f"✅ Partições salvas em {os.path.abspath(caminho)}")

    def executar_cross_validation(self, X_tfidf, y, label_columns, caminho='particoes.pkl'):
        if not os.path.exists(caminho):
            self.gerar_particoes_multilabel(X_tfidf, y, caminho=caminho)

        with open(caminho, 'rb') as f:
            folds = pickle.load(f)

        for i, (train_idx, test_idx) in enumerate(folds):
            logging.info(f"\n========================= 📚 Fold {i+1}/{len(folds)} =========================")
            X_train, y_train = X_tfidf[train_idx], y[train_idx]
            X_test, y_test = X_tfidf[test_idx], y[test_idx]
            self.train_and_predict(X_train, y_train, X_test, y_test, label_columns)

    def run(self, file_path, usar_cross_validation=False):
        X, y_df = self.load_and_prepare_data(file_path)
        if X is None or y_df is None:
            return

        y_df = self.filter_labels(y_df)
        label_columns = y_df.columns.tolist()
        logging.info(f"✅ Rótulos mantidos após filtragem: {label_columns}")

        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)
        y = y_df.values

        if usar_cross_validation:
            self.executar_cross_validation(X_tfidf, y, label_columns)
        else:
            X_train, y_train, X_test, y_test = iterative_train_test_split(X_tfidf, y, test_size=0.3)
            logging.info(f"📐 Partições: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
            self.train_and_predict(X_train, y_train, X_test, y_test, label_columns)
            

if __name__ == "__main__":
    hierarchy = {
        'Hate.speech': {
            'R': ['Hate.speech', 'No.hate.speech'],
            'Hate.speech': ['Homophobia', 'Racism', 'Sexism', 'Body', 'Ideology', 'Religion', 'Migrants', 'OtherLifestyle', 'Origin'],
            'Homophobia': ['Homossexuals'],           
            'Homossexuals': ['Gays', 'Lesbians'],
            'Racism': ['Black.people'],
            'Ideology': ['Left.wing.ideology', 'Feminists'],            
            'Sexism': ['Women', 'Men', 'Feminists', 'Transexuals'],
            'Transexuals': ['Trans.women'],
            'Women': ['Trans.women', 'Fat.women', 'Ugly.women', 'Lesbians'],            
            'Body': ['Fat.people', 'Ugly.people'],
            'Ugly.people': ['Ugly.women'],
            'Fat.people': ['Fat.women'],
            'Migrants': ['Immigrants', 'Refugees'],
            'Religion': ['Islamists', 'Muslims']            
        }
    }

    classifier = HierarchicalLocalPerNodeClassifier(hierarchy=hierarchy)
    classifier.run("2019-05-28_portuguese_hate_speech_hierarchical_classification.csv")
