import re
import random
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertAdam
import numpy as np
import torch.nn.functional as F
from sklearn import metrics,preprocessing
from sklearn.preprocessing import LabelEncoder

PAD, CLS, SEP = '[PAD]', '[CLS]' ,'[SEP]'


def get_shift_prob(label):
    similarities = np.load('./similarity.npy').tolist()
    dept = np.load('./dept_id.npy').tolist()
    index = dept.index(label)
    prob_dict = {}
    for i in range(0,len(similarities[index])):
        if i == 0:
            prob_dict[dept[int(similarities[index][i][0]) - 1]] = 1
        else:
            prob_dict[dept[int(similarities[index][i][0])-1]] = similarities[index][i][1]
    sum = 0
    for key in prob_dict.keys():
        if key != label:
            sum += prob_dict[key]
    for key in prob_dict.keys():
        if key != label:
            prob_dict[key] /= sum
            prob_dict[key] *= 0
    return prob_dict

def roulette(prob_dict:dict):
    random_val = random.random()
    probability = 0
    for key in prob_dict.keys():
        probability += prob_dict[key]
        if probability >= random_val:
            return key
        else:
            continue


class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = './train_kn.txt'  # 训练集
        self.dev_path =  './dev.txt'  # 验证集
        self.test_path = './test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            './class.txt',encoding='utf-8').readlines()]  # 类别名单
        self.save_path ='./' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 4  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './mcbert'  # 预训练模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        labels = []
        features = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                line = re.sub(re.compile(r'[\n]'),'',line)
                # 读数据，去除首尾空格，分离标签与句子内容
                content, label = line.split()
                labels.append(label)
                features.append(content)
            le = LabelEncoder()
            le.fit(config.class_list)
            labels = le.transform(labels)
            
            for i in range(len(features)):
                # 使用配置中的tokenize对句子内容进行分割，句首增加'[CLS]'
                token = config.tokenizer.tokenize(features[i])
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(labels[i]), seq_len, mask))
        return contents,le

    train,le = load_dataset(config.train_path, config.pad_size)
    dev,_ = load_dataset(config.dev_path, config.pad_size)
    test,_ = load_dataset(config.test_path, config.pad_size)
    return train, dev, test,le


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    if torch.cuda.device_count() > 1: #查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
	    model = torch.nn.DataParallel(model)#多gpu训练,自动选择gpu
    model.to(device)
    model.train()  # model.train()将启用BatchNormalization和Dropout，相应的，model.eval()则不启用BatchNormalization和Dropout
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_acc = -float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            label_list = labels.tolist()
            new_label = []
            for label in label_list:
                shift_label = le.inverse_transform([label])[0]
                prob_dict = get_shift_prob(shift_label)
                shift_label = roulette(prob_dict)
                shift_label = le.transform([shift_label])[0]
                new_label.append(shift_label)
            new_label = torch.tensor(new_label).to(config.device)
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, new_label)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                print(msg.format(total_batch, F.cross_entropy(outputs, labels), train_acc, dev_loss, dev_acc),improve)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':
    dataset_path = ''

    config = Config(dataset_path)  # 初始化配置
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 固定随机因子


    print("Loading data...")
    train_data, dev_data, test_data,le = build_dataset(config)  # 数据集预处理
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)


    # train
    model = Model(config).to(config.device)  # 确定训练设备
    train(config, model, train_iter, dev_iter, test_iter)  # 开始训练