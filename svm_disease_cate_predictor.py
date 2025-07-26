import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from underthesea import word_tokenize


# Tai dataset
def dataset_loader(path):
    data = pd.read_csv(path)
    return data

# Ma hoa label su dung LabelEncoder
def encoder(data, column_name, encoded_label_colname, ecr):
    data[encoded_label_colname] = ecr.fit_transform(data[column_name])
    return data

# Kiem tra va truc quan hoa dataset
def check_dataset_info(data, label_colname):
    print("Nam dong dau tien dataset: ")
    print(data.sample(5))
    print("\nSo luong label: ", data[label_colname].nunique())
    print("\nThong tin dataset: ")
    print(data.info())

def visualize_dataset(data, name):
    sns.countplot(x=name, data=data)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Lay stopword tu github
response = requests.get("https://github.com/stopwords/vietnamese-stopwords/blob/a453d389e1b52e20748ca83ddd8b0faebb04f5aa/vietnamese-stopwords.txt")
vietnamese_stopwords = set(response.text.splitlines())

# Xu ly van ban
def process_text(text):
    if isinstance(text, str):
        # Loai bo ky tu dac biet
        text = re.sub(r'[^\w\s]', '', text)
        # Tach tu su dung underthesea
        text_tokenized = word_tokenize(text, format="text")
        # Loai bo cac stop word
        words = text_tokenized.split()
        filtered_words = [word for word in words if word.lower() not in vietnamese_stopwords]
        return " ".join(filtered_words)
    return ""

# Xu ly feature dataset
def process_data(data, label_colname, feature_colname, processed_feature_colname):
    # Loai bo cac dong trung lap
    data.drop_duplicates(inplace=True)
    # Loai bo cac dong co du lieu thieu
    data = data.dropna(subset=[feature_colname, label_colname])
    # Xu ly va tach tu o cot feature
    data[processed_feature_colname] = data[feature_colname].apply(process_text)
    return data

# Vectorize feature bang TfidfVectorizer
def vectorizer_data(vtr, data, processed_feature_colname=None):
    if processed_feature_colname is not None:
        vector = vtr.fit_transform(data[processed_feature_colname])
    else:
        vector = vtr.fit_transform(data)
    return vector


# Huan luyen mo hinh
def load_model(X, Y):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
    class_weights_dict = dict(zip(np.unique(Y), class_weights))
    md = SVC(kernel='linear', C=1.0, class_weight=class_weights_dict, random_state=42)
    md.fit(X, Y)
    return md

def validate_model(md,encoder, X, y):
    y_pred = md.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1_weighted = f1_score(y, y_pred, average='weighted')
    f1_macro = f1_score(y, y_pred, average='macro')
    class_names = label_encoder.classes_
    class_report = classification_report(y_test, y_pred, target_names=class_names)

    print("\n" + "-"*30)
    print("Hiệu suất của model:")
    print("Độ chính xác: ", accuracy)
    print("F1 Score (Weighted): ", f1_weighted)
    print("F1 Score (Macro): ", f1_macro)
    print("\nBáo cáo của nhóm bệnh:")
    print(class_report)

    print("\nClassification Report:\n", classification_report(y, y_pred, target_names=encoder.classes_))

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Nhóm bệnh dự đoán')
    plt.ylabel('Nhóm bệnh thực tế')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Du doan nhom benh dua tren dau vao
def predict(input_data, md, vtr, ecr):
    ip_processed = process_text(input_data)
    ip_vec = vtr.transform([ip_processed])
    predicted_label_index = md.predict(ip_vec)[0]
    predicted = ecr.inverse_transform([predicted_label_index])[0]
    return predicted

def n_predict(input_data, md, vtr, ecr, n=3):
    ip_processed = process_text(input_data)
    ip_vec = vtr.transform([ip_processed])
    probs = md.predict_proba(ip_vec)[0]
    top_n_idx = np.argsort(probs)[::-1][:n]
    results = []
    for idx in top_n_idx:
        label = ecr.inverse_transform([idx])[0]
        prob = probs[idx]
        results.append((label, prob))
    return results

def show_result(input_data, result):
    print(f"Câu hỏi: {input_data}")
    print(f"Dự đoán bệnh: {result}")
    print("-" * 30)

if __name__ == "__main__":
    # Tai va chuan bi du lieu huan luyen
    df = dataset_loader("hf://datasets/joon985/ViMedical_Disease_Category/ViMedicalDiseaseCategory.csv")
    check_dataset_info(df, 'DiseaseCategory')
    visualize_dataset(df, 'DiseaseCategory')
    label_encoder = LabelEncoder()
    df = encoder(df, 'DiseaseCategory', 'DiseaseCategory_encoded', label_encoder)
    df = process_data(df, 'DiseaseCategory', 'Question', 'Question_processed')

    # Chia tap train, validation va test theo ty le 8:2
    X_train, X_test, y_train, y_test = train_test_split(
        df["Question_processed"].tolist(),
        df["DiseaseCategory_encoded"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["DiseaseCategory_encoded"].tolist()
    )

    # Vectorize du lieu
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer_data(vectorizer,X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Huan luyen mo hinh
    model = load_model(X_train_vec, y_train)

    # Kiem tra mo hinh
    validate_model(model, label_encoder, X_test_vec, y_test)

