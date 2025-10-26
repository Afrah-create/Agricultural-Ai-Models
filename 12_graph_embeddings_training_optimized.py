import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

print("Loading Knowledge Graph...")

with open('/content/drive/MyDrive/Final/data/processed/unified_knowledge_graph.json', 'r') as f:
    unified_triples = json.load(f)

def prepare_graph_data(unified_triples, max_triples=10000):
    print(f"Preparing {min(len(unified_triples), max_triples)} triples...")
    
    if len(unified_triples) > max_triples:
        unified_triples = unified_triples[:max_triples]
    
    entities = set()
    relations = set()
    
    for triple in unified_triples:
        entities.add(triple['subject'])
        entities.add(triple['object'])
        relations.add(triple['predicate'])
    
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
    relation_to_id = {relation: idx for idx, relation in enumerate(sorted(relations))}
    id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}
    id_to_relation = {idx: relation for relation, idx in relation_to_id.items()}
    
    triples_indices = []
    for triple in unified_triples:
        head = entity_to_id[triple['subject']]
        relation = relation_to_id[triple['predicate']]
        tail = entity_to_id[triple['object']]
        triples_indices.append([head, relation, tail])
    
    triples_indices = np.array(triples_indices)
    
    print(f"Entities: {len(entities)}, Relations: {len(relations)}, Triples: {len(triples_indices)}")
    
    return triples_indices, entity_to_id, relation_to_id, id_to_entity, id_to_relation

def create_negative_samples(triples, num_entities, num_samples_per_triple=1):
    negatives = []
    for triple in triples:
        for _ in range(num_samples_per_triple):
            corrupt_head = np.random.rand() > 0.5
            neg_triple = triple.copy()
            if corrupt_head:
                neg_triple[0] = np.random.randint(0, num_entities)
            else:
                neg_triple[2] = np.random.randint(0, num_entities)
            negatives.append(neg_triple)
    return np.array(negatives)

def create_train_val_test_split(triples_indices, test_size=0.2, val_size=0.1):
    train_val_indices, test_indices = train_test_split(
        triples_indices, test_size=test_size, random_state=42
    )
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size/(1-test_size), random_state=42
    )
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = F.normalize(self.entity_embeddings(head), p=2, dim=1)
        r = F.normalize(self.relation_embeddings(relation), p=2, dim=1)
        t = F.normalize(self.entity_embeddings(tail), p=2, dim=1)
        
        score = torch.norm(h + r - t, p=2, dim=1)
        return score
    
    def predict(self, head, relation, tail):
        with torch.no_grad():
            score = self.forward(head, relation, tail)
            return torch.sigmoid(-score)

class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(DistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        score = torch.sum(h * r * t, dim=1)
        return score
    
    def predict(self, head, relation, tail):
        with torch.no_grad():
            score = self.forward(head, relation, tail)
            return torch.sigmoid(score)

class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(ComplEx, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings_real.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)
    
    def forward(self, head, relation, tail):
        h_real = self.entity_embeddings_real(head)
        h_imag = self.entity_embeddings_imag(head)
        r_real = self.relation_embeddings_real(relation)
        r_imag = self.relation_embeddings_imag(relation)
        t_real = self.entity_embeddings_real(tail)
        t_imag = self.entity_embeddings_imag(tail)
        
        score = torch.sum(
            h_real * r_real * t_real +
            h_imag * r_real * t_imag +
            h_real * r_imag * t_imag -
            h_imag * r_imag * t_real,
            dim=1
        )
        return score
    
    def predict(self, head, relation, tail):
        with torch.no_grad():
            score = self.forward(head, relation, tail)
            return torch.sigmoid(score)

class GCN(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, hidden_dim=200):
        super(GCN, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.gcn1 = nn.Linear(embedding_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        h_gcn = F.relu(self.gcn1(h))
        h_gcn = self.gcn2(h_gcn)
        
        t_gcn = F.relu(self.gcn1(t))
        t_gcn = self.gcn2(t_gcn)
        
        score = torch.sum(h_gcn * r * t_gcn, dim=1)
        return score
    
    def predict(self, head, relation, tail):
        with torch.no_grad():
            score = self.forward(head, relation, tail)
            return torch.sigmoid(score)

class GraphSAGE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, hidden_dim=200):
        super(GraphSAGE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.sage1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.sage2 = nn.Linear(hidden_dim, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        h_concat = torch.cat([h, r], dim=1)
        h_sage = F.relu(self.sage1(h_concat))
        h_sage = self.sage2(h_sage)
        
        t_concat = torch.cat([t, r], dim=1)
        t_sage = F.relu(self.sage1(t_concat))
        t_sage = self.sage2(t_sage)
        
        score = torch.sum(h_sage * t_sage, dim=1)
        return score
    
    def predict(self, head, relation, tail):
        with torch.no_grad():
            score = self.forward(head, relation, tail)
            return torch.sigmoid(score)

def train_model(model, train_data, val_data, num_epochs=50, learning_rate=0.001, device='cpu', margin=1.0):
    print(f"Training {model.__class__.__name__}...")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        batch_size = 128
        num_batches = 0
        
        pbar = tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for i in pbar:
            batch = train_data[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            head = torch.tensor(batch[:, 0], dtype=torch.long).to(device)
            relation = torch.tensor(batch[:, 1], dtype=torch.long).to(device)
            tail = torch.tensor(batch[:, 2], dtype=torch.long).to(device)
            
            neg_samples = create_negative_samples(batch, model.num_entities, num_samples_per_triple=1)
            neg_head = torch.tensor(neg_samples[:, 0], dtype=torch.long).to(device)
            neg_relation = torch.tensor(neg_samples[:, 1], dtype=torch.long).to(device)
            neg_tail = torch.tensor(neg_samples[:, 2], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            pos_scores = model(head, relation, tail)
            neg_scores = model(neg_head, neg_relation, neg_tail)
            
            if isinstance(model, TransE):
                target = torch.ones(len(pos_scores)).to(device)
                loss = F.margin_ranking_loss(pos_scores, neg_scores, target, margin=margin)
            else:
                pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                loss = (pos_loss + neg_loss) / 2
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                if len(batch) < 2:
                    continue
                
                head = torch.tensor(batch[:, 0], dtype=torch.long).to(device)
                relation = torch.tensor(batch[:, 1], dtype=torch.long).to(device)
                tail = torch.tensor(batch[:, 2], dtype=torch.long).to(device)
                
                neg_samples = create_negative_samples(batch, model.num_entities, num_samples_per_triple=1)
                neg_head = torch.tensor(neg_samples[:, 0], dtype=torch.long).to(device)
                neg_relation = torch.tensor(neg_samples[:, 1], dtype=torch.long).to(device)
                neg_tail = torch.tensor(neg_samples[:, 2], dtype=torch.long).to(device)
                
                pos_scores = model(head, relation, tail)
                neg_scores = model(neg_head, neg_relation, neg_tail)
                
                if isinstance(model, TransE):
                    target = torch.ones(len(pos_scores)).to(device)
                    loss = F.margin_ranking_loss(pos_scores, neg_scores, target, margin=margin)
                else:
                    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                    loss = (pos_loss + neg_loss) / 2
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_losses

def evaluate_model(model, test_data, device='cpu'):
    print(f"Evaluating {model.__class__.__name__}...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    batch_size = 128
    
    pbar = tqdm(range(0, len(test_data), batch_size), desc="Evaluation", leave=False)
    
    with torch.no_grad():
        for i in pbar:
            batch = test_data[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            head = torch.tensor(batch[:, 0], dtype=torch.long).to(device)
            relation = torch.tensor(batch[:, 1], dtype=torch.long).to(device)
            tail = torch.tensor(batch[:, 2], dtype=torch.long).to(device)
            
            pos_scores = model.predict(head, relation, tail)
            
            neg_samples = create_negative_samples(batch, model.num_entities, num_samples_per_triple=1)
            neg_head = torch.tensor(neg_samples[:, 0], dtype=torch.long).to(device)
            neg_relation = torch.tensor(neg_samples[:, 1], dtype=torch.long).to(device)
            neg_tail = torch.tensor(neg_samples[:, 2], dtype=torch.long).to(device)
            
            neg_scores = model.predict(neg_head, neg_relation, neg_tail)
            
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores)
            ])
            
            all_predictions.extend(all_scores.cpu().numpy())
            all_targets.extend(all_labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    predictions_binary = (predictions > 0.5).astype(int)
    
    try:
        roc_auc = roc_auc_score(targets, predictions)
    except:
        roc_auc = 0.5
    
    try:
        avg_precision = average_precision_score(targets, predictions)
    except:
        avg_precision = 0.0
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions_binary),
        'precision': precision_score(targets, predictions_binary, zero_division=0),
        'recall': recall_score(targets, predictions_binary, zero_division=0),
        'f1_score': f1_score(targets, predictions_binary, zero_division=0),
        'roc_auc': roc_auc,
        'average_precision': avg_precision
    }
    
    return metrics

print("Starting Training...")

triples_indices, entity_to_id, relation_to_id, id_to_entity, id_to_relation = prepare_graph_data(
    unified_triples, max_triples=10000
)

train_indices, val_indices, test_indices = create_train_val_test_split(triples_indices)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

models = {
    'TransE': TransE(len(entity_to_id), len(relation_to_id), embedding_dim=100),
    'DistMult': DistMult(len(entity_to_id), len(relation_to_id), embedding_dim=100),
    'ComplEx': ComplEx(len(entity_to_id), len(relation_to_id), embedding_dim=100),
    'GCN': GCN(len(entity_to_id), len(relation_to_id), embedding_dim=100, hidden_dim=200),
    'GraphSAGE': GraphSAGE(len(entity_to_id), len(relation_to_id), embedding_dim=100, hidden_dim=200)
}

models_results = {}

for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    
    train_losses, val_losses = train_model(
        model, train_indices, val_indices,
        num_epochs=50, learning_rate=0.001, device=device
    )
    
    print("\nEvaluating on all splits...")
    train_metrics = evaluate_model(model, train_indices, device=device)
    val_metrics = evaluate_model(model, val_indices, device=device)
    test_metrics = evaluate_model(model, test_indices, device=device)
    
    models_results[model_name] = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    print(f"\n{model_name} Performance:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
    print("-" * 50)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        train_val = train_metrics[metric]
        val_val = val_metrics[metric]
        test_val = test_metrics[metric]
        print(f"{metric:<20} {train_val:<12.4f} {val_val:<12.4f} {test_val:<12.4f}")

results_to_save = {}
for model_name, results in models_results.items():
    results_to_save[model_name] = {
        'train_losses': results['train_losses'],
        'val_losses': results['val_losses'],
        'train_metrics': results['train_metrics'],
        'val_metrics': results['val_metrics'],
        'test_metrics': results['test_metrics']
    }

with open('/content/drive/MyDrive/Final/data/processed/graph_embedding_results.json', 'w') as f:
    json.dump(results_to_save, f, indent=2)

print(f"\nResults saved to: /content/drive/MyDrive/Final/data/processed/graph_embedding_results.json")

# Save trained models
print("\nSaving trained models...")
models_dir = '/content/drive/MyDrive/Final/data/processed/trained_models'

# Create models directory if it doesn't exist
import os
os.makedirs(models_dir, exist_ok=True)

# Save all trained models
for model_name, results in models_results.items():
    model_path = os.path.join(models_dir, f'{model_name.lower()}_model.pth')
    torch.save(results['model'].state_dict(), model_path)
    print(f"  ✅ {model_name} saved to: {model_path}")

# Save model metadata (entity/relation mappings)
metadata = {
    'entity_to_id': entity_to_id,
    'relation_to_id': relation_to_id,
    'id_to_entity': id_to_entity,
    'id_to_relation': id_to_relation,
    'num_entities': len(entity_to_id),
    'num_relations': len(relation_to_id),
    'embedding_dim': 100  # From the model initialization
}

metadata_path = os.path.join(models_dir, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ Model metadata saved to: {metadata_path}")

# Identify and save the best model
print("\nIdentifying best performing model...")
best_model_name = None
best_f1_score = -1

for model_name, results in models_results.items():
    f1_score = results['test_metrics']['f1_score']
    if f1_score > best_f1_score:
        best_f1_score = f1_score
        best_model_name = model_name

if best_model_name:
    print(f"🏆 Best model: {best_model_name} (F1: {best_f1_score:.4f})")
    
    # Save best model with special naming
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    torch.save(models_results[best_model_name]['model'].state_dict(), best_model_path)
    
    # Save best model info
    best_model_info = {
        'model_name': best_model_name,
        'f1_score': best_f1_score,
        'all_metrics': models_results[best_model_name]['test_metrics'],
        'model_path': best_model_path,
        'metadata_path': metadata_path
    }
    
    best_model_info_path = os.path.join(models_dir, 'best_model_info.json')
    with open(best_model_info_path, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"  ✅ Best model saved to: {best_model_path}")
    print(f"  ✅ Best model info saved to: {best_model_info_path}")
else:
    print("❌ No best model identified")

print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")
print(f"\n{'Model':<15} {'Split':<10} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
print("-" * 70)

for model_name in models_results.keys():
    for split_name, split_key in [('Train', 'train_metrics'), ('Val', 'val_metrics'), ('Test', 'test_metrics')]:
        metrics = models_results[model_name][split_key]
        print(f"{model_name:<15} {split_name:<10} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f} {metrics['roc_auc']:<8.4f}")
    print("-" * 70)

print(f"\nBest Test Set Performance:")
for metric in ['accuracy', 'f1_score', 'roc_auc']:
    best_model = max(models_results.keys(), key=lambda x: models_results[x]['test_metrics'][metric])
    best_score = models_results[best_model]['test_metrics'][metric]
    print(f"{metric}: {best_model} ({best_score:.4f})")