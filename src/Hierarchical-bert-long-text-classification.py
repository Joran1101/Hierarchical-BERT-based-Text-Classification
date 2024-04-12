# -*- coding: utf-8 -*-
# @Time    : 2024/4/8 21:40
# @Author  : xjw
# @FileName: Hierarchical-bert-long-text-classification.py
# @Software: PyCharm
# @Blog    ：https://github.com/Joran1101
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

# 示例数据
texts = [
    "这是一段很长的文本,包含了多个段落。<paragraph_begin>第一段内容是关于自然语言处理的介绍。</paragraph_end><paragraph_begin"
    ">第二段内容是关于深度学习模型的应用。</paragraph_end><paragraph_begin>第三段内容是关于注意力机制的解释。</paragraph_end>",
    "这是另一段长文本,讲述了一个故事情节。<paragraph_begin>第一段是故事的开头,介绍了主人公。</paragraph_end><paragraph_begin>第二段是故事的发展,"
    "描述了主人公遇到的困难。</paragraph_end><paragraph_begin>第三段是故事的结尾,主人公最终解决了问题。</paragraph_end>"
]
# 示例标签更新为10个标签的情况，这里只是示例，实际上每个段落都应有一个长度为10的标签数组
labels = [
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
]


class LabelAttention(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super(LabelAttention, self).__init__()
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, hidden_dim))

    def forward(self, doc_embeddings):
        # 计算标签和文档之间的相似度
        similarity = torch.matmul(doc_embeddings, self.label_embeddings.t())
        # 直接计算注意力权重
        attention_weights = torch.softmax(similarity, dim=1)
        # 应用注意力权重到标签嵌入上
        attended_label_embeddings = torch.matmul(attention_weights, self.label_embeddings)
        # 将文档嵌入和加权的标签嵌入合并
        doc_representation = doc_embeddings + attended_label_embeddings
        return doc_representation


# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512, num_segments=3):
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_len = max_len
        self.num_segments = num_segments
        self.segments, self.attention_masks, self.segment_labels = self.process_texts(texts, labels)

    def process_texts(self, texts, labels):
        segments, attention_masks, segment_labels = [], [], []
        for text, label in zip(texts, labels):
            paragraphs = text.split("<paragraph_end><paragraph_begin>")
            processed_paragraphs, processed_masks = [], []
            for paragraph in paragraphs:
                # 应用滑动窗口方法处理每个段落
                sub_paragraphs, sub_masks = self.apply_sliding_window(paragraph)
                processed_paragraphs.extend(sub_paragraphs)
                processed_masks.extend(sub_masks)

            # 确保处理后的数据不超过预定的段落数量
            processed_paragraphs = processed_paragraphs[:self.num_segments]
            processed_masks = processed_masks[:self.num_segments]

            # 如果段落不够，用0填充
            while len(processed_paragraphs) < self.num_segments:
                processed_paragraphs.append([0] * self.max_len)
                processed_masks.append([0] * self.max_len)

            segments.append(processed_paragraphs)
            attention_masks.append(processed_masks)
            segment_labels.append(label)

        return segments, attention_masks, segment_labels

    def apply_sliding_window(self, paragraph, window_size=512, stride=256):
        # 对段落进行编码，不截断也不填充
        encoded_dict = self.tokenizer.encode_plus(paragraph, add_special_tokens=True, return_attention_mask=True,
                                                  truncation=False, padding=False)
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        # 应用滑动窗口
        sub_paragraphs, sub_masks = [], []
        for i in range(0, len(input_ids), stride):
            sub_input_ids = input_ids[i:i + window_size]
            sub_attention_mask = attention_mask[i:i + window_size]

            # 用0填充到window_size
            padding_length = window_size - len(sub_input_ids)
            sub_input_ids.extend([0] * padding_length)
            sub_attention_mask.extend([0] * padding_length)

            sub_paragraphs.append(sub_input_ids)
            sub_masks.append(sub_attention_mask)

        return sub_paragraphs, sub_masks

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = torch.tensor(self.segments[idx], dtype=torch.long)
        mask = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        label = torch.tensor(self.segment_labels[idx], dtype=torch.float)
        return {
            'input_ids': segment,
            'attention_mask': mask,
            'labels': label
        }


class TransformerAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_labels):
        super(TransformerAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, num_labels)

    def forward(self, x, mask=None):
        aggregated_output = self.transformer_encoder(x, src_key_padding_mask=mask)
        # 使用平均池化代替取第一个段落的输出
        pooled_output = torch.mean(aggregated_output, dim=0)
        logits = self.output_layer(pooled_output)
        return logits


# 定义模型
class HierarchicalModel(nn.Module):
    def __init__(self, num_labels):
        super(HierarchicalModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.label_attention = LabelAttention(self.bert.config.hidden_size, num_labels)
        self.aggregator = TransformerAggregator(embed_dim=self.bert.config.hidden_size, num_heads=8, num_layers=3,
                                                num_labels=num_labels)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        batch_size, num_segments, seq_len = input_ids.size()

        # 重塑input_ids和attention_mask以适应bert模型的输入
        input_ids = input_ids.view(-1, seq_len)  # [batch_size * num_segments, seq_len]
        attention_mask = attention_mask.view(-1, seq_len)  # [batch_size * num_segments, seq_len]

        # 通过BERT模型
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # 获取[CLS]标记的输出作为每个段落的表示
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size * num_segments, hidden_size]

        # 重塑回原来的batch_size和num_segments维度
        cls_embeddings = cls_embeddings.view(batch_size, num_segments, -1)  # [batch_size, num_segments, hidden_size]

        # 应用标签注意力机制到每个段落的表示上
        doc_representation = torch.zeros(batch_size, self.bert.config.hidden_size, device=input_ids.device)
        for i in range(batch_size):
            # 假设cls_embeddings的形状为[batch_size, num_segments, hidden_size]
            # 假设我们想要对每个文档的所有段落的表示求平均，然后应用标签注意力
            doc_embedding = cls_embeddings[i].mean(dim=0)  # 对段落求平均，结果形状[hidden_size]
            doc_embedding = doc_embedding.unsqueeze(0)  # 增加批次维度，以匹配LabelAttention的期望输入形状[1, hidden_size]
            attended_doc_embedding = self.label_attention(doc_embedding)  # 应用标签注意力
            # 确保doc_representation[i]能够接收attended_doc_embedding的输出
            doc_representation[i] = attended_doc_embedding.squeeze(0)  # 移除批次维度，形状变回[hidden_size]

        # 应用Transformer聚合器
        # 假设doc_representation的形状为[batch_size, hidden_size]
        # 添加一个假的序列长度维度，使其形状变为[1, batch_size, hidden_size]
        doc_representation = doc_representation.unsqueeze(0)

        # 现在可以正确地调用self.aggregator，因为输入形状符合预期
        logits = self.aggregator(doc_representation)

        return logits


def compute_class_weights(labels):
    # 计算每个类的正样本数
    positive_counts = labels.sum(0)
    # 使用逆频率进行加权
    class_weights = 1.0 / (positive_counts + 1e-5)  # 避免除以零，添加一个小的常数

    # 权重归一化
    class_weights /= class_weights.sum()  # 确保所有权重的和为1

    return class_weights


# 在主程序中计算类别权重
labels_tensor = torch.tensor(labels, dtype=torch.float)
class_weights = compute_class_weights(labels_tensor)


# 训练函数中使用BCEWithLogitsLoss
def train(model, data_loader, scheduler, device, class_weights, num_epochs=3):
    model.train()  # 确保模型处于训练模式
    class_weights = class_weights.to(device)

    for epoch in range(num_epochs):  # 添加支持多个训练周期
        total_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = loss_fn(outputs, labels)
            loss.backward()
            scheduler.step()

            total_loss += loss.item()

            # 打印每个批次的损失
            print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(data_loader)
        # 打印每轮训练结束后的平均损失
        print(f'Epoch: {epoch + 1}, Average Training loss: {avg_loss:.4f}\n')


# 主程序
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = HierarchicalModel(num_labels=10).to(device)

train_texts, val_texts, train_labels, val_labels = texts, [], labels, []  # 假设没有验证集
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

scheduler = CosineAnnealingLR(optimizer, T_max=100)  # 或者 StepLR(optimizer, step_size=30, gamma=0.1) 用于阶梯式下降

# 训练模型
train(model, train_loader, scheduler, device, class_weights, num_epochs=10)

MODEL_SAVE_PATH = "./model/model.pth"  # 确保目录和文件名都在路径中

# 训练结束后保存模型
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

"""------------------------------------------------------------------------------------"""
# 创建模型实例
model = HierarchicalModel(num_labels=10)

# 加载模型权重
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

# 将模型转移到评估模式
model.eval()

# 确保模型在正确的设备上
model.to(device)

# 测试数据
test_texts = [
    "这是一个测试文本，包含了一些关于自然语言处理的介绍。",
    "另一个测试文本，讲述了深度学习的应用。"
]

# 假设测试标签（在实际测试中可能不需要标签，除非你要评估模型性能）
test_labels = [[1, 0, 0, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0]]

# 创建测试数据集
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

# 创建DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 存储所有预测结果
predictions = []

# 禁用梯度计算
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 进行预测
        logits = model(input_ids, attention_mask)

        # 计算概率
        probs = F.sigmoid(logits)

        predictions.append(probs.cpu().numpy())

# 打印预测结果
for i, text in enumerate(test_texts):
    print(f"文本: {text}")
    print(f"预测概率: {predictions[i]}")
    print("---")
