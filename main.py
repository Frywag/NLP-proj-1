import os
import jieba
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置环境变量 (可选，先尝试保守设置)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:3072'


class CustomTextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length, stop_words_file=None):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.label_map = {}
        self.stop_words = self._load_stop_words(stop_words_file) if stop_words_file else set()
        self._load_data()

    def _load_stop_words(self, stop_words_file):
        stop_words = set()
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            for line in f:
                stop_words.add(line.strip())
        return stop_words

    def _clean_text(self, text):
        # 去除网址
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 去除 @ 用户名
        text = re.sub(r'@\w+', '', text)
        # 去除多余的空格,并转为半角
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_data(self):
        temp_labels = []
        temp_data = []

        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('_!_')
                if len(parts) != 5:
                    continue

                news_id, category_code, category_name, title, keywords = parts
                title = self._clean_text(title)
                title = ''.join(
                    char for char in title if char.isalnum() or char in ['，', '。', '？', '！']
                )
                title = ' '.join([word for word in jieba.cut(title) if word not in self.stop_words])
                temp_data.append(title)
                temp_labels.append(category_code)

        label_set = sorted(list(set(temp_labels)))
        for i, label in enumerate(label_set):
            self.label_map[label] = i

        for i, label in enumerate(temp_labels):
                self.data.append(temp_data[i])
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }


class BertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)  # 可以尝试不同的 dropout 值
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] 对应的输出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_and_evaluate(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, label_map, accumulation_steps=4):
    """
    训练和评估模型。

    Args:
        model: 要训练的模型。
        train_dataloader: 训练数据的 DataLoader。
        val_dataloader: 验证数据的 DataLoader。
        device: 训练设备 ('cuda' 或 'cpu')。
        num_epochs: 训练的 epoch 数。
        learning_rate: 学习率。
        label_map: 标签映射字典

    Returns:
        train_losses: 训练损失列表。
        val_losses: 验证损失列表。
        val_accuracies: 验证准确率列表。
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0  # 初始化最佳验证准确率

    # 构建类别名称列表
    label_names = [None] * len(label_map)
    for name, idx in label_map.items():
        label_names[idx] = name

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()  # 零梯度放在循环开头

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss = loss / accumulation_steps  # 将损失除以累积步数

            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # 清零梯度

            total_train_loss += loss.item() * accumulation_steps

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                total_val_accuracy += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        avg_val_accuracy = total_val_accuracy / len(val_dataloader.dataset)
        val_accuracies.append(avg_val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_accuracy:.4f}")

        # 计算并打印其他评估指标
        print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

        # 保存最佳模型
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Best model saved!")

    return train_losses, val_losses, val_accuracies


def plot_results(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_map):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_map.keys(),
        yticklabels=label_map.keys(),
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    # 超参数设置
    pretrained_model_name = './bert-base-chinese'  # BERT 模型路径
    max_length = 25  # 根据数据集调整
    batch_size = 16  # 根据 GPU 内存调整
    num_epochs = 10  # 适当增加, 以充分训练
    learning_rate = 1e-5  # 可以尝试不同的学习率
    filepath = 'D:/UserDesktop/55/yuyanchuli/toutiao_cat_data.txt'  # 数据文件路径!
    stop_words_file = 'D:/UserDesktop/55/yuyanchuli/stopwords_Chinese.txt'  # 停用词文件路径!

    # 加载 tokenizer 和数据集
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    dataset = CustomTextDataset(filepath, tokenizer, max_length, stop_words_file)
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42, stratify=dataset.labels
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    label_map = dataset.label_map

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification(pretrained_model_name, num_labels=len(label_map))
    model.to(device)

    # 训练和评估
    train_losses, val_losses, val_accuracies = train_and_evaluate(
        model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, label_map, accumulation_steps=4
    )

    # 可视化结果
    plot_results(train_losses, val_losses, val_accuracies)

    # 加载最佳模型并进行最终评估
    best_model = BertForSequenceClassification(pretrained_model_name, num_labels=len(label_map))
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.to(device)
    best_model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:  # *应该使用单独的测试集*
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = best_model(input_ids, attention_mask)
            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    plot_confusion_matrix(all_labels, all_preds, label_map)