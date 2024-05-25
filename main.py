import torch.nn.functional as F

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset_builder
from torch.utils.data import Dataset

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.metrics import ndcg_score

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data['query'])
    
    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data.keys()}

def load_data(path, batch_size=32):
    #positive_samples
    qrels_dev = pd.read_json(path+'qrels_dev.jsonl', lines=True).reset_index(drop=True)
    qrels_test = pd.read_json(path+'qrels_test.jsonl', lines=True).reset_index(drop=True)
    positive_samples = pd.concat([qrels_dev, qrels_test], ignore_index=True)
    
    corpus = pd.read_json(path+'corpus.jsonl', lines=True).reset_index(drop=True)
    queries = pd.read_json(path+'queries.jsonl', lines=True).reset_index(drop=True)
    
    pair = defaultdict(list)
    for query_id, corpus_id in zip(positive_samples['query-id'], positive_samples['corpus-id']):
        pair[query_id].append(corpus_id)
    
    corpus_id_to_index = {corpus_id: index for index, corpus_id in enumerate(corpus['_id'])}
    index_to_corpus_id = {index: corpus_id for index, corpus_id in enumerate(corpus['_id'])}
    query_id_to_index = {query_id: index for index, query_id in enumerate(queries['_id'])}
    index_to_query_id = {index: query_id for index, query_id in enumerate(queries['_id'])}
    dataset = {'query': [queries['text'][query_id_to_index[query_id]] for query_id in positive_samples['query-id']], 'corpus': [corpus['text'][corpus_id_to_index[corpus_id]] for corpus_id in positive_samples['corpus-id']], 
               'query_id': positive_samples['query-id'], 'corpus_id': positive_samples['corpus-id']}

    
    train_query, test_query, train_corpus, test_corpus, train_query_id, test_query_id, train_corpus_id, test_corpus_id = train_test_split(
       dataset['query'], dataset['corpus'], dataset['query_id'].tolist(), dataset['corpus_id'].tolist(), test_size=0.2)

    train_data = {'query': train_query, 'corpus': train_corpus, 'query_id': train_query_id, 'corpus_id': train_corpus_id}
    train_dataset = MyDataset(train_data)
    test_query, val_query, test_corpus, val_corpus, test_query_id, val_query_id, test_corpus_id, val_corpus_id = train_test_split(
         test_query, test_corpus, test_query_id, test_corpus_id, test_size=0.5)
    
    test_data = {'query': test_query, 'corpus': test_corpus, 'query_id': test_query_id, 'corpus_id': test_corpus_id}
    test_dataset = MyDataset(test_data)
    val_data = {'query': val_query, 'corpus': val_corpus, 'query_id': val_query_id, 'corpus_id': val_corpus_id}
    val_dataset = MyDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, positive_samples, train_loader, val_loader, test_loader, query_id_to_index, corpus_id_to_index, train_data, val_data, test_data, pair

def tokenize_data(data, tokenizer, device):
    embeddings = tokenizer(data, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    return embeddings
    

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def train(model, tokenizer, data, train_loader, val_loader, train_data, val_data, num_epochs, learning_rate, patience, pairs, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_val_loss = float('inf')
    no_improvement = 0

    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            
            batch_size = len(batch['query'])
            inputs = prepend_prefix(batch)
            inputs = tokenize_data(inputs, tokenizer, device)
            inputs_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model(inputs_ids)
            embeddings = average_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            qk = embeddings[:batch_size] @ embeddings[batch_size:].T
            kq = qk.T
            scores_matrix = torch.cat([qk, kq], dim= 0)
            labels = torch.arange(batch_size).to(device)
            labels = torch.cat([labels, labels], dim=0)
            loss = criterion(scores_matrix/0.1, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            
        avg_loss = total_loss / len(train_loader)
        t_acc = accuracy(train_data, model, tokenizer, pairs, device)
        train_loss.append(float(avg_loss))
        train_acc.append(float(t_acc))
        v_loss, v_acc = evaluate(model, val_loader, val_data, pairs, device)
        val_loss.append(float(v_loss))
        val_acc.append(float(v_acc))
        print(f'End of Epoch {epoch}, Training Loss: {avg_loss}, Training Accuracy: {t_acc}, Validation Loss: {v_loss}, Validation Accuracy: {v_acc}')
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            no_improvement = 0
            best_model = model.state_dict()
        else:
            no_improvement += 1
            if no_improvement > patience:
                print(f"Early stopping at epoch {epoch} with best validation loss {best_val_loss}")
                model.load_state_dict(best_model)
                break

    torch.save(model.state_dict(), 'model.pth')
    return train_loss, train_acc, val_loss, val_acc


def evaluate(model, test_loader, test_data, pairs, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_size = len(batch['query'])
            inputs = prepend_prefix(batch)
            inputs = tokenize_data(inputs, tokenizer, device)
            inputs_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model(inputs_ids)
            embeddings = average_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            qk = embeddings[:batch_size] @ embeddings[batch_size:].T
            kq = qk.T
            scores_matrix = torch.cat([qk, kq], dim= 0)
            labels = torch.arange(batch_size).to(device)
            labels = torch.cat([labels, labels], dim=0)
            loss = criterion(scores_matrix/0.1, labels)
            test_loss += loss.item()
    avg_loss = test_loss / len(test_loader)
    acc = accuracy(test_data, model, tokenizer, pairs, device)
    
    return avg_loss, acc


def prepend_prefix(data):
    queries = data['query']
    corpuses = data['corpus']
    queries = ['query: ' + query for query in queries]
    corpus = ['passage: ' + passage for passage in corpuses]
    return queries+corpus

    

def accuracy(data, model, tokenizer, pairs, device):
    with torch.no_grad():
        model.eval()
        query_encode_embeddings = []
        corpus_encode_embeddings = []
        batch_size = 16
        total_score = 0
        query_list = data['query_id']
        corpus = data['corpus_id']
        n_queries = len(query_list)
        count = 0
        for i in range(0, len(data['query']), batch_size):
            queries = data['query'][i:i+batch_size]
            
            inputs = prepend_prefix({'query': queries, 'corpus': data['corpus'][i:i+batch_size]})
            inputs = tokenize_data(inputs, tokenizer, device)
            inputs_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            with torch.cuda.amp.autocast():
                outputs = model(inputs_ids)
                embeddings = average_pool(outputs.last_hidden_state, attention_mask)
                embeddings = F.normalize(embeddings, p=2, dim=1)
              
            size = embeddings.size(0)//2
            scores_matrix = embeddings[:size] @ embeddings[size:].T  
            query_encode_embeddings.append(embeddings[:size])
            corpus_encode_embeddings.append(embeddings[size:])
            score = ndcg_score(np.eye(size).astype(int), scores_matrix.cpu().numpy(), k=10)
            total_score += score
            count += 1
        score = total_score / count
  
        query_encode_embeddings = torch.cat(query_encode_embeddings, dim=0)
        query_encode_embeddings = F.normalize(query_encode_embeddings, p=2, dim=1)
        corpus_encode_embeddings = torch.cat(corpus_encode_embeddings, dim=0)
        corpus_encode_embeddings = F.normalize(corpus_encode_embeddings, p=2, dim=1)
        
        scores_matrix = query_encode_embeddings @ corpus_encode_embeddings.T
        score1 = ndcg_score(np.eye(len(data['query'])).astype(int), scores_matrix.cpu().numpy(), k=10)
    

    return score1



def plot_loss(train_loss, train_acc, val_loss, val_acc):
    train_loss = torch.tensor(train_loss).cpu().numpy()
    train_acc = torch.tensor(train_acc).cpu().numpy()
    val_loss = torch.tensor(val_loss).cpu().numpy()
    val_acc = torch.tensor(val_acc).cpu().numpy()
    
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    # Add legend
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

    plt.clf()
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy(nDCG@10)")
    plt.savefig("accuracy.png")
    
    
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current Device: {}".format(device))
    
    epochs = 30
    learning_rate = 1e-5
    batch_size = 16
    patience = 10
    
    
    data_path = 'Data/'
    dataset, positive_samples, train_loader, val_loader, test_loader, query_id_to_index, corpus_id_to_index, train_data, val_data, test_data, pairs = load_data(data_path, batch_size)
    
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    model = AutoModel.from_pretrained('intfloat/e5-base-v2')
    
    model.to(device)
    loss, acc = evaluate(model, test_loader, test_data, pairs, device)
    
    print(f"Average Test Loss: {loss}, Test Accuracy: {acc}")
    
    print("Starting Training")
    
    train_loss, train_acc, val_loss, val_acc = train(model, tokenizer, dataset, train_loader, val_loader, train_data, val_data, epochs, learning_rate, patience, pairs, device)
    
    plot_loss(train_loss, train_acc, val_loss, val_acc)
    
    print("Starting Evaluation")
    
    loss, acc = evaluate(model, test_loader, test_data, pairs, device)
    
    print(f"Average Test Loss: {loss}, Test Accuracy: {acc}")
    
