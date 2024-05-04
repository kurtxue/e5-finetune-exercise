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

import matplotlib.pyplot as plt

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
    
    corpus_id_to_index = {corpus_id: index for index, corpus_id in enumerate(corpus['_id'])}
    index_to_corpus_id = {index: corpus_id for index, corpus_id in enumerate(corpus['_id'])}
    query_id_to_index = {query_id: index for index, query_id in enumerate(queries['_id'])}
    index_to_query_id = {index: query_id for index, query_id in enumerate(queries['_id'])}
    dataset = {'query': [queries['text'][query_id_to_index[query_id]] for query_id in positive_samples['query-id']], 'corpus': [corpus['text'][corpus_id_to_index[corpus_id]] for corpus_id in positive_samples['corpus-id']], 
               'query_id': positive_samples['query-id'], 'corpus_id': positive_samples['corpus-id']}

    
    train_query, test_query, train_corpus, test_corpus, train_query_id, test_query_id, train_corpus_id, test_corpus_id = train_test_split(
       dataset['query'], dataset['corpus'], dataset['query_id'].tolist(), dataset['corpus_id'].tolist(), test_size=0.2)

    train_dataset = MyDataset({'query': train_query, 'corpus': train_corpus, 'query_id': train_query_id, 'corpus_id': train_corpus_id})
    test_query, val_query, test_corpus, val_corpus, test_query_id, val_query_id, test_corpus_id, val_corpus_id = train_test_split(
         test_query, test_corpus, test_query_id, test_corpus_id, test_size=0.5)
    
    test_dataset = MyDataset({'query': test_query, 'corpus': test_corpus, 'query_id': test_query_id, 'corpus_id': test_corpus_id})
    val_dataset = MyDataset({'query': val_query, 'corpus': val_corpus, 'query_id': val_query_id, 'corpus_id': val_corpus_id})


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, positive_samples, train_loader, val_loader, test_loader, query_id_to_index, corpus_id_to_index, train_dataset, val_dataset, test_dataset

def tokenize_data(data, tokenizer, query_id_to_index, corpus_id_to_index, device):
    queries = list(set(data['query']))
    corpus = list(set(data['corpus']))
    input_texts = prepend_prefix(queries, corpus)
    embeddings = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    return embeddings
    
def generate_negative_samples(batch):
    negative_index = {'negative_q': [], 'negative_c': []}
    positive_index = []
    queries = batch['query_id'].unique()
    corpuses = batch['corpus_id'].unique()
    
    for i in range(len(batch['query_id'])):
        index = [0, 0]
        negative_q = []
        negative_c = []
        query = batch['query_id'][i]
        corpus = batch['corpus_id'][i]
        unrelated_samples = {'query_id': [], 'corpus_id': []}
        for j,_ in enumerate(batch):
            if batch['query_id'][j] != query:
                unrelated_samples['query_id'].append(batch['query_id'][j])
            if batch['corpus_id'][j] != corpus:
                unrelated_samples['corpus_id'].append(batch['corpus_id'][j])
        for x, q in enumerate(queries):
            if q != query and q in unrelated_samples['query_id']:
                negative_c.append(x)
            if q == query:
                index[0] = x
        for y, c in enumerate(corpuses):
            if  c != corpus and c in unrelated_samples['corpus_id']:
                negative_q.append(y)
            if  c == corpus:
                index[1] = y
        positive_index.append(tuple(index))
        if negative_c == []:
            negative_index['negative_c'].append(0)
        negative_index['negative_q'].append([(index[0], i) for i in negative_q])
        negative_index['negative_c'].append([(i, index[1]) for i in negative_c])
        

                    
    return positive_index, negative_index, queries, corpuses

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def train(model, tokenizer, data, train_loader, val_loader, train_data, val_data, num_epochs, learning_rate, patience, query_id_to_index, corpus_id_to_index, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            positive_index, negative_index, queries, corpuses = generate_negative_samples(batch)
            inputs = tokenize_data(batch, tokenizer, query_id_to_index, corpus_id_to_index, device)
            n_queries = len(queries)
            n_corpuses = len(corpuses)
            inputs_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model(inputs_ids)
            embeddings = average_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores_matrix = (embeddings[:n_queries] @ embeddings[n_queries:].T) * 100
            loss = compute_loss(scores_matrix, positive_index, negative_index)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        t_acc = accuracy(train_data, model, tokenizer, query_id_to_index, corpus_id_to_index, device)
        train_loss.append(float(avg_loss))
        train_acc.append(float(t_acc))
        v_loss, v_acc = evaluate(model, val_loader, val_data, query_id_to_index, corpus_id_to_index, device)
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


def evaluate(model, test_loader, test_data, query_id_to_index, corpus_id_to_index, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            positive_index, negative_index, queries, corpuses = generate_negative_samples(batch)
            inputs = tokenize_data(batch, tokenizer, query_id_to_index, corpus_id_to_index, device)
            n_queries = len(queries)
            n_corpuses = len(corpuses)
            inputs_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model(inputs_ids)
            embeddings = average_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores_matrix = (embeddings[:n_queries] @ embeddings[n_queries:].T) * 100
            loss = compute_loss(scores_matrix, positive_index, negative_index)
            test_loss += loss.item()
    avg_loss = test_loss / len(test_loader)
    acc = accuracy(test_data, model, tokenizer, query_id_to_index, corpus_id_to_index, device)
    
    return avg_loss, acc


def prepend_prefix(queries, corpuses):
    queries = ['query: ' + query for query in queries]
    corpus = ['passage: ' + passage for passage in corpuses]
    return queries+corpus
    
    
def compute_loss(scores_matrix, positive_index, negative_index):
    total_loss = 0
    for pos, neg_q, neg_c in zip(positive_index, negative_index['negative_q'], negative_index['negative_c']):
        q_rows, q_cols = zip(*neg_q)
    
        loss_q = -torch.log(torch.exp(scores_matrix[pos]) / (torch.exp(scores_matrix[pos]) + torch.sum(torch.exp(scores_matrix[q_rows, q_cols]))))
        if neg_c == 0:
            loss_c = 0   
        else:
            c_rows, c_cols = zip(*neg_c)
            loss_c = -torch.log(torch.exp(scores_matrix[pos]) / (torch.exp(scores_matrix[pos]) + torch.sum(torch.exp(scores_matrix[c_rows, c_cols]))))
        total_loss += loss_q + loss_c
    avg_loss = total_loss / len(positive_index)
    return avg_loss
    
    
def accuracy(data, model, tokenizer, query_id_to_index, corpus_id_to_index, device):
    total_score = 0
    queries = data['query_id'].unique()
    n_queries = len(queries)
    inputs = tokenize_data(data, tokenizer, query_id_to_index, corpus_id_to_index, device)
    inputs_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model(inputs_ids)
    embeddings = average_pool(outputs.last_hidden_state, attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores_matrix = (embeddings[:n_queries] @ embeddings[n_queries:].T) * 100
    
    for query in queries:
        query_index = query_id_to_index[query]
        predicted_top10_indices = torch.argsort(scores_matrix[query_index], descending=True)[:10]
        df = pd.DataFrame(data)
        true_related = df[df['query_id'] == query].shape[0]
        true_top10_score = [1] * true_related + [0] * 10 
        true_top10_score = true_top10_score[:10]
        dcg = np.sum( (2 ** scores_matrix[query_index][predicted_top10_indices] - 1) / np.log2(np.arange(2, 12)))
        idcg = np.sum( (2 ** true_top10_score - 1) / np.log2(np.arange(2, 12)))
        ndcg = dcg / idcg
        total_score += ndcg
    score = total_score / len(queries)
    return score



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
    
    epochs = 10
    learning_rate = 1e-4
    batch_size = 32
    
    
    
    data_path = 'Data/'
    dataset, positive_samples, train_loader, val_loader, test_loader, query_id_to_index, corpus_id_to_index, train_data, val_data, test_data = load_data(data_path, batch_size)
    
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    model = AutoModel.from_pretrained('intfloat/e5-base-v2')
    
    model.to(device)
    
    print("Starting Training")
    
    train_loss, train_acc, val_loss, val_acc = train(model, tokenizer, dataset, train_loader, val_loader, train_data, val_data, epochs, learning_rate, 10, query_id_to_index, corpus_id_to_index, device)
    
    plot_loss(train_loss, train_acc, val_loss, val_acc)
    
    print("Starting Evaluation")
    
    loss, acc = evaluate(model, test_loader, test_data, query_id_to_index, corpus_id_to_index, device)
    
    print(f"Average Test Loss: {loss}, Test Accuracy: {acc}")
    
