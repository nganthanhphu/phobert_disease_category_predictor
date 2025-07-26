import random
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from underthesea import word_tokenize
from huggingface_hub import hf_hub_download

# Tao seed de tao ket qua co the lap lai
seed = 48
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Kiem tra GPU va thiet lap device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng: {device}")

# Bat cudnn de toi uu hoa toc do fine tune bang GPU va tat benchmark bat deterministic de tao ket qua lap lai
if device.type == "cuda":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

# Xu ly van ban
def process_text(text):
    if isinstance(text, str):
        # Loai bo ky tu dac biet
        text = re.sub(r'[^\w\s]', '', text)
        # Tach tu su dung underthesea
        text_tokenized = word_tokenize(text, format="text")
        return text_tokenized
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

# Hien thi ket qua du doan
def show_result(input_data, result):
    print(f"Câu hỏi: {input_data}")
    print(f"Dự đoán bệnh: {result}")
    print("-" * 30)

# Cau hinh mo hinh PhoBERT
PHOBERT_MODEL = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

# Encode bang PhoBERT tokenizer
def phobert_encode(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Tao PhoBERT classifier
class PhoBERTClassifier(torch.nn.Module):
    def __init__(self, labels_num):
        super(PhoBERTClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained(PHOBERT_MODEL)
        self.classifier = torch.nn.Linear(self.phobert.config.hidden_size, labels_num)

    def forward(self, input_ids, attention_mask):
        output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(output.last_hidden_state[:, 0, :])
        return logits

# Lop Dataset tuy chinh cho viec huan luyen PhoBERT
class PhoBERTDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_length=128):
        self.encodings = phobert_encode(x, tokenizer, max_length)
        self.labels = torch.tensor(y)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Fine tune model PhoBERT
def train_phobert_classifier(X_train, y_train, X_val, y_val, labels_num, tokenizer, epochs=8, batch_size=8):
    train_ds = PhoBERTDataset(X_train, y_train, tokenizer)
    model = PhoBERTClassifier(labels_num)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = PhoBERTDataset(X_val, y_val, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    print(f"Train bằng {device} với {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        if val_loader:
            model.eval()
            preds, true_label = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].cpu().numpy()
                    logits = model(input_ids, attention_mask)
                    pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
                    preds.extend(pred_idx)
                    true_label.extend(labels)
            val_accuracy = accuracy_score(true_label, preds)
            val_f1_weighted = f1_score(true_label, preds, average='weighted')
            val_f1_macro = f1_score(true_label, preds, average='macro')

            print(f"Độ chính xác tập validation: {val_accuracy:.4f}")
            print(f"F1 Score (Weighted) tập validation: {val_f1_weighted:.4f}")
            print(f"F1 Score (Macro) tập validation: {val_f1_macro:.4f}")
            model.train()

    return model

def get_model_path(repo_id, filename):
    print("Tải mô hình từ Hugging Face")
    return hf_hub_download(repo_id=repo_id, filename=filename)

#Load model PhoBERT da huan luyen
def load_phobert_model(model_path, labels_num):
    model = PhoBERTClassifier(labels_num)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

# Test model
def test_model_using_test_dataset(model, tokenizer, label_encoder, X_test, y_test):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for text in X_test:
            inputs = phobert_encode(text, tokenizer)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            pred_idx = torch.argmax(logits, dim=1).item()
            y_pred.append(pred_idx)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_
    class_report = classification_report(y_test, y_pred, target_names=class_names)

    print("\n" + "-"*30)
    print("Hiệu suất trên tập test của model:")
    print(f"Độ chính xác: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print("\nBáo cáo của nhóm bệnh:")
    print(class_report)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Phân loại nhóm bệnh dựa trên câu hỏi triệu chứng')
    plt.xlabel('Nhóm bệnh dự đoán')
    plt.ylabel('Nhóm bệnh thực tế')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Du doan su dung PhoBERT da duoc fine tune
def predict(x, model, tokenizer, label_encoder):
    model.eval()
    with torch.no_grad():
        inputs = phobert_encode(x, tokenizer)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        pred_idx = torch.argmax(logits, dim=1).item()
        return label_encoder.inverse_transform([pred_idx])[0]

def n_predict(x, model, tokenizer, label_encoder, top_k=3):
    model.eval()
    with torch.no_grad():
        inputs = phobert_encode(x, tokenizer)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        probs = torch.nn.functional.softmax(logits, dim=1)
        topk_prob, topk_idx = torch.topk(probs, k=top_k, dim=1)
        topk_prob = topk_prob.squeeze().cpu().numpy()
        topk_idx = topk_idx.squeeze().cpu().numpy()
        topk_labels = label_encoder.inverse_transform(topk_idx)
        return tuple(zip(topk_labels, topk_prob))

if __name__ == "__main__":
    # Tai va chuan bi du lieu huan luyen
    df = dataset_loader("hf://datasets/joon985/ViMedical_Disease_Category/ViMedicalDiseaseCategory.csv")
    check_dataset_info(df, 'DiseaseCategory')
    visualize_dataset(df, 'DiseaseCategory')
    label_encoder = LabelEncoder()
    df = encoder(df, 'DiseaseCategory', 'DiseaseCategory_encoded', label_encoder)
    df = process_data(df, 'DiseaseCategory', 'Question', 'Question_processed')

    # Chia tap train, validation va test theo ty le 8:1:1
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["Question_processed"].tolist(),
        df["DiseaseCategory_encoded"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["DiseaseCategory_encoded"].tolist()
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # Fine tune PhoBERT model
    num_labels = len(label_encoder.classes_)
    phobert_model = train_phobert_classifier(X_train, y_train, X_val, y_val, num_labels, tokenizer, epochs=16, batch_size=16)

    # Luu model
    torch.save(phobert_model.state_dict(), "model.pth")

    # Tai model PhoBERT da fine tune (neu can)
    # num_labels = len(label_encoder.classes_)
    # phobert_model = load_phobert_model(get_model_path("joon985/PhoBERTDiseaseCategoryPredictor", "model.pth"), num_labels)

    # Danh gia model tren tap test
    test_model_using_test_dataset(phobert_model, tokenizer, label_encoder, X_test, y_test)
