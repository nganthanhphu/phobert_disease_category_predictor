import random
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
def check_dataset_info(data):
    print("Nam dong dau tien dataset: ")
    print(data.sample(5))
    print("\nSo luong label: ", data['Disease'].nunique())
    print("\nThong tin dataset: ")
    print(data.info())

def visualize_dataset(data, name):
    sns.countplot(x=name, data=data)
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

def show_n_result(input_data, result):
    print(f"Câu hỏi: {input_data}")
    print("Các bệnh mà bạn có thể mắc phải (sắp xếp theo xác suất giảm dần):")
    for (i, res) in enumerate(result):
        print(f"Bệnh {i+1}: {res}")
    print("-" * 30)

# Ghi ket qua du doan ra file txt
def export_result_to_txt(input_data, result, file):
    file.write(f"Câu hỏi: {input_data}\n")
    file.write(f"Dự đoán bệnh: {result}\n")
    file.write("-" * 30 + "\n")

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

    print(f"Train bằng {device} với {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y_train = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(logits, y_train)
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
                    y_train = batch['labels'].cpu().numpy()
                    logits = model(input_ids, attention_mask)
                    pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
                    preds.extend(pred_idx)
                    true_label.extend(y_train)
            print(f"Độ chính xác: {accuracy_score(true_label, preds):.4f}")
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

def test_model_using_dataset(model, tokenizer, label_encoder, dataset, feature_colname, processed_feature_colname, output_path="test_results.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for _,row in dataset.iterrows():
            question = row[feature_colname]
            processed_question = row[processed_feature_colname]
            result = predict(processed_question, model, tokenizer, label_encoder)
            export_result_to_txt(question, result, f)
    print("Kết quả đã được ghi vào file.")

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
        topk_idx = topk_idx.squeeze().cpu().numpy()
        topk_labels = label_encoder.inverse_transform(topk_idx)
        return topk_labels.tolist()

if __name__ == "__main__":
    # Tai va chuan bi du lieu huan luyen
    df = dataset_loader("hf://datasets/PB3002/ViMedical_Disease/ViMedical_Disease.csv")
    check_dataset_info(df)
    visualize_dataset(df, 'Disease')
    label_encoder = LabelEncoder()
    df = encoder(df, 'Disease', 'Disease_encoded', label_encoder)
    df = process_data(df, 'Disease', 'Question', 'Question_processed')

    # Chia tap train va tap validation ty le 8:2
    X_train, X_val, y_train, y_val = train_test_split(
        df["Question_processed"].tolist(),
        df["Disease_encoded"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["Disease_encoded"].tolist()
    )

    # Fine tune PhoBERT model
    num_labels = len(label_encoder.classes_)
    phobert_model = train_phobert_classifier(X_train, y_train, X_val, y_val, num_labels, tokenizer, epochs=16, batch_size=16)

    # Luu model
    torch.save(phobert_model.state_dict(), "model.pth")

    # Tai model PhoBERT da fine tune
    # num_labels = len(label_encoder.classes_)
    # phobert_model = load_phobert_model(get_model_path("joon985/PhoBERTDiseasePredictor", "model.pth"), num_labels)

    # Kiem thu model
    df_test = dataset_loader("hf://datasets/joon985/MedicalTestDataset/MedicalTestDataset.csv")
    df_test = process_data(df_test, 'Disease', 'Question', 'Question_processed')
    test_model_using_dataset(phobert_model, tokenizer, label_encoder, df_test, 'Question','Question_processed')
