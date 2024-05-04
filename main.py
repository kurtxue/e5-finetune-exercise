import torch.nn.functional as F

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



def load_data(path, batch_size=32):
    #positive_samples
    qrels_dev = pd.read_json(path+'qrels_dev.jsonl', lines=True)
    qrels_test = pd.read_json(path+'qrels_test.jsonl', lines=True)
    positive_samples = pd.concat([qrels_dev, qrels_test], ignore_index=True)
    
    corpus = pd.read_json(path+'corpus.jsonl', lines=True)
    queries = pd.read_json(path+'queries.jsonl', lines=True)
    
    corpus_id_to_index = {corpus_id: index for index, corpus_id in enumerate(corpus['corpus_id'])}
    index_to_corpus_id = {index: corpus_id for index, corpus_id in enumerate(corpus['corpus_id'])}
    query_id_to_index = {query_id: index for index, query_id in enumerate(queries['query_id'])}
    index_to_query_id = {index: query_id for index, query_id in enumerate(queries['query_id'])}
    true_scores = [0 for _ in range(len(corpus)) for _ in range(len(queries))]
    for index, row in positive_samples.iterrows():
        query_id = row['query_id']
        corpus_id = row['corpus_id']
        true_scores[query_id_to_index[query_id]][corpus_id_to_index[corpus_id]] = row['score']
    dataset = {'query': [queries['query'][query_id_to_index[query_id]] for query_id in positive_samples['query_id']], 'corpus': [corpus['corpus'][corpus_id_to_index[corpus_id]] for corpus_id in positive_samples['corpus_id']], 
               'query_id': positive_samples['query_id'], 'corpus_id': positive_samples['corpus_id'], 'score': positive_samples['score']}
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    test_dataset, val_dataset = train_test_split(test_dataset, test_size=0.5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataset, positive_samples, train_loader, val_loader, test_loader, true_scores, query_id_to_index

def tokenize_data(data, tokenizer):
    queries = data['query'].unique()   
    corpus = data['corpus'].unique()
    input_texts = prepend_prefix(queries, corpus)
    embeddings = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    return embeddings
    
def generate_negative_samples(batch):
    negative_index = {'negative_q': [], 'negative_c': []}
    positive_index = []
    queries = batch['query_id'].unique()
    corpuses = batch['corpus_id'].unique()
   
    for row in enumerate(batch):
        index
        negative_q = []
        negative_c = []
        query = row['query_id']
        corpus = row['corpus_id']
        unrelated_samples = {'query_id': [], 'corpus_id': []}
        for row in enumerate(batch):
            if row['query_id'] != query and row['corpus_id'] != corpus:
                unrelated_samples['query_id'].append(row['query_id'])
                unrelated_samples['corpus_id'].append(row['corpus_id'])
        for x, q in enumerate(queries):
            if q != query and q not in unrelated_samples['query_id']:
                negative_c.append(x)
            for y, c in enumerate(corpuses):
                if  c != corpus and c not in unrelated_samples['corpus_id']:
                    negative_q.append(y)
                if q == query and c == corpus:
                    index = (x, y)
        positive_index.append(index)
        negative_index['negative_q'].append([(index[0], i) for i in negative_q])
        negative_index['negative_c'].append([(i, index[1]) for i in negative_c])
                    
    return positive_index, negative_index, queries, corpuses

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def train(model, tokenizer, data, train_loader, val_loader, num_epochs, learning_rate, patience, true_scores, query_id_to_index, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch.to(device)
            positive_index, negative_index, queries, corpuses = generate_negative_samples(batch)
            inputs = prepend_prefix(queries, corpuses)
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
        t_acc = accuracy(train_loader, true_scores, query_id_to_index)
        train_loss.append(float(avg_loss))
        train_acc.append(float(t_acc))
        v_loss, v_acc = evaluate(model, val_loader, true_scores, query_id_to_index)
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


def evaluate(model, test_loader, true_scores, query_id_to_index):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            positive_index, negative_index, queries, corpuses = generate_negative_samples(batch)
            inputs = prepend_prefix(queries, corpuses)
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
    acc = accuracy(test_loader, true_scores, query_id_to_index)
    
    return avg_loss, acc


def prepend_prefix(queries, corpuses):
    queries = ['query: ' + query for query in queries]
    corpus = ['passage: ' + passage for passage in corpuses]
    return queries+corpus
    
    
def compute_loss(scores_matrix, positive_index, negative_index):
    total_loss = 0
    for i, (pos, neg) in enumerate(zip(positive_index, negative_index)):
        loss_q = -torch.log(torch.exp(scores_matrix[pos]) / (torch.exp(scores_matrix[pos]) + torch.sum(torch.exp(scores_matrix[neg['negative_q']]))))
        loss_c = -torch.log(torch.exp(scores_matrix[pos]) / (torch.exp(scores_matrix[pos]) + torch.sum(torch.exp(scores_matrix[neg['negative_c']]))))
        total_loss += loss_q + loss_c
    avg_loss = total_loss / len(positive_index)
    return avg_loss
    
    
def accuracy(data, true_scores, query_id_to_index):
    total_score = 0
    input = prepend_prefix(data['query'], data['corpus'])
    
    embeddings = model(input)
    embeddings = average_pool(embeddings.last_hidden_state, data['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    predicted_scores = (embeddings[:2] @ embeddings[2:].T) * 100
    queries = data['query_id'].unique()
    
    for query in queries:
        query_index = query_id_to_index[query]
        predicted_top10_indices = torch.argsort(predicted_scores[query_index], descending=True)[:10]
        true_top10_indices = torch.argsort(true_scores[query_index], descending=True)[:10]
        dcg = np.sum( (2 ** predicted_scores[query_index][predicted_top10_indices] - 1) / np.log2(np.arange(2, 12)))
        idcg = np.sum( (2 ** true_scores[query_index][true_top10_indices] - 1) / np.log2(np.arange(2, 12)))
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
    learning_rate = 1e-5
    batch_size = 32
    
    
    
    data_path = 'Data/'
    dataset, positive_samples, train_loader, val_loader, test_loader, true_scores, query_id_to_index = load_data(data_path, batch_size)
    
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    model = AutoModel.from_pretrained('intfloat/e5-base-v2')
    
    model.to(device)
    tokenizer.to(device)
    
    print("Starting Training")
    
    train_loss, train_acc, val_loss, val_acc = train(model, tokenizer, dataset, train_loader, val_loader, epochs, learning_rate, 3, true_scores, query_id_to_index, device)
    
    plot_loss(train_loss, train_acc, val_loss, val_acc)
    
    print("Starting Evaluation")
    
    loss, acc = evaluate(model, test_loader, true_scores, query_id_to_index)
    
    print(f"Average Test Loss: {loss}, Test Accuracy: {acc}")
    
