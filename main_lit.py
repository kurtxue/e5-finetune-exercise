import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from collections import defaultdict
from datasets import load_dataset_builder
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data['query'])
    
    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}

def load_data(path, batch_size=32):
    qrels_dev = pd.read_json(path+'qrels_dev.jsonl', lines=True).reset_index(drop=True)
    qrels_test = pd.read_json(path+'qrels_test.jsonl', lines=True).reset_index(drop=True)
    positive_samples = pd.concat([qrels_dev, qrels_test], ignore_index=True)
    
    corpus = pd.read_json(path+'corpus.jsonl', lines=True).reset_index(drop=True)
    queries = pd.read_json(path+'queries.jsonl', lines=True).reset_index(drop=True)
    
    
    corpus_id_to_index = {corpus_id: index for index, corpus_id in enumerate(corpus['_id'])}
    query_id_to_index = {query_id: index for index, query_id in enumerate(queries['_id'])}
    
    dataset = {
        'query': [queries['text'][query_id_to_index[query_id]] for query_id in positive_samples['query-id']],
        'corpus': [corpus['text'][corpus_id_to_index[corpus_id]] for corpus_id in positive_samples['corpus-id']],
        'query_id': positive_samples['query-id'],
        'corpus_id': positive_samples['corpus-id']
    }
    
    train_query, test_query, train_corpus, test_corpus, train_query_id, test_query_id, train_corpus_id, test_corpus_id = train_test_split(
        dataset['query'], dataset['corpus'], dataset['query_id'].tolist(), dataset['corpus_id'].tolist(), test_size=0.2
    )
    
    train_data = {'query': train_query, 'corpus': train_corpus, 'query_id': train_query_id, 'corpus_id': train_corpus_id}
    train_dataset = MyDataset(train_data)
    
    test_query, val_query, test_corpus, val_corpus, test_query_id, val_query_id, test_corpus_id, val_corpus_id = train_test_split(
        test_query, test_corpus, test_query_id, test_corpus_id, test_size=0.5
    )
    
    test_data = {'query': test_query, 'corpus': test_corpus, 'query_id': test_query_id, 'corpus_id': test_corpus_id}
    test_dataset = MyDataset(test_data)
    val_data = {'query': val_query, 'corpus': val_corpus, 'query_id': val_query_id, 'corpus_id': val_corpus_id}
    val_dataset = MyDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, positive_samples, train_loader, val_loader, test_loader, query_id_to_index, corpus_id_to_index, train_data, val_data, test_data

def tokenize_data(data, tokenizer, device):
    embeddings = tokenizer(data, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    return embeddings
    

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def prepend_prefix(data):
    queries = data['query']
    corpuses = data['corpus']
    queries = ['query: ' + query for query in queries]
    corpus = ['passage: ' + passage for passage in corpuses]
    return queries + corpus

def ndcg_score_custom(data, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        query_encode_embeddings = []
        corpus_encode_embeddings = []
        count = 0
        for batch in data:
            batch_size = len(batch['query'])
            inputs = prepend_prefix(batch)
            inputs = tokenize_data(inputs, tokenizer, device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model(input_ids)
            embeddings = average_pool(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            scores_matrix = embeddings[:batch_size] @ embeddings[batch_size:].T
            query_encode_embeddings.append(embeddings[:batch_size])
            corpus_encode_embeddings.append(embeddings[batch_size:])
            count += batch_size

  
        query_encode_embeddings = torch.cat(query_encode_embeddings, dim=0)
        query_encode_embeddings = F.normalize(query_encode_embeddings, p=2, dim=1)
        corpus_encode_embeddings = torch.cat(corpus_encode_embeddings, dim=0)
        corpus_encode_embeddings = F.normalize(corpus_encode_embeddings, p=2, dim=1)
        
        scores_matrix = query_encode_embeddings @ corpus_encode_embeddings.T
        score = ndcg_score(np.eye(count), scores_matrix.cpu().numpy(), k=10)
    return score

class QueryCorpusModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.val_outputs = []

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        return outputs


    
    def training_step(self, batch, batch_idx):
        inputs = prepend_prefix(batch)
        inputs = tokenize_data(inputs, self.tokenizer, self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self(input_ids)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        batch_size = len(batch['query'])
        qk = embeddings[:batch_size] @ embeddings[batch_size:].T
        kq = qk.T
        scores_matrix = torch.cat([qk, kq], dim=0)
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        loss = self.criterion(scores_matrix/0.1, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = prepend_prefix(batch)
        inputs = tokenize_data(inputs, self.tokenizer, self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self(input_ids)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        batch_size = len(batch['query'])
        qk = embeddings[:batch_size] @ embeddings[batch_size:].T
        kq = qk.T
        scores_matrix = torch.cat([qk, kq], dim=0)
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        loss = self.criterion(scores_matrix/0.1, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.val_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        val_acc = ndcg_score_custom(self.trainer.val_dataloaders, self.model, self.tokenizer, self.device)
        self.log('val_acc', val_acc, prog_bar=True)
        print(f"Validation Accuracy: {val_acc}")
        self.val_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def test_step(self, batch, batch_idx):
        pass 
    
    def on_test_end(self):
        test_acc = ndcg_score_custom(self.trainer.test_dataloaders, self.model, self.tokenizer, self.device)
        print(f"Test Accuracy: {test_acc}")
        return test_acc
    
    


def plot_loss(train_loss, val_loss, val_acc):
    train_loss = torch.tensor(train_loss).cpu().numpy()
    # train_acc = torch.tensor(train_acc).cpu().numpy()
    val_loss = torch.tensor(val_loss).cpu().numpy()
    val_acc = torch.tensor(val_acc).cpu().numpy()
    
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

    plt.clf()
    plt.figure()
    # plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy(nDCG@10)")
    plt.savefig("accuracy.png")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current Device: {}".format(device))
    
    epochs = 1
    learning_rate = 1e-5
    batch_size = 16
    patience = 10
    
    data_path = 'Data/'
    dataset, positive_samples, train_loader, val_loader, test_loader, query_id_to_index, corpus_id_to_index, train_data, val_data, test_data = load_data(data_path, batch_size)
    
    model = QueryCorpusModel(learning_rate)
    
    model.to(device)
    
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[pl.callbacks.EarlyStopping(monitor='avg_val_loss', patience=patience)])
    
    trainer.fit(model, train_loader, val_loader)
    
    print("Starting Evaluation")
    trainer.test(model=model, dataloaders=test_loader)
    
    print(trainer.logged_metrics)
    # Save loss and accuracy plots
    plot_loss(trainer.logged_metrics['train_loss'], trainer.logged_metrics['val_loss'], trainer.logged_metrics['train_acc'], trainer.logged_metrics['val_acc'])
