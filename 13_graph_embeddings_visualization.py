"""
Phase 5, Cell 2: Graph Embeddings Visualization
This cell creates comprehensive visualizations for all trained graph embedding models
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Loading Graph Embedding Results...")

# Load results from the comprehensive training
try:
    with open('/content/drive/MyDrive/Final/data/processed/graph_embedding_results.json', 'r') as f:
        results = json.load(f)
    print(f"✅ Results loaded for {len(results)} models")
    print(f"Models found: {list(results.keys())}")
except Exception as e:
    print(f"❌ Error loading results: {e}")
    print("Please run the graph embeddings training cell first")
    
    # Create mock results based on your training output for demonstration
    print("Creating mock results based on training output...")
    results = {
        'TransE': {
            'train_losses': [0.7488, 0.5153, 0.4519, 0.4232, 0.4013, 0.3841, 0.3786, 0.3660, 0.3656, 0.3620],
            'val_losses': [0.6034, 0.5166, 0.4980, 0.4677, 0.4664, 0.4373, 0.4465, 0.4208, 0.4437, 0.4425],
            'train_metrics': {
                'accuracy': 0.5000,
                'precision': 0.0000,
                'recall': 0.0000,
                'f1_score': 0.0000,
                'roc_auc': 0.1246,
                'average_precision': 0.0561
            },
            'val_metrics': {
                'accuracy': 0.5000,
                'precision': 0.0000,
                'recall': 0.0000,
                'f1_score': 0.0000,
                'roc_auc': 0.1214,
                'average_precision': 0.0550
            },
            'test_metrics': {
                'accuracy': 0.5000,
                'precision': 0.0000,
                'recall': 0.0000,
                'f1_score': 0.0000,
                'roc_auc': 0.1214,
                'average_precision': 0.0550
            }
        },
        'DistMult': {
            'train_losses': [0.6902, 0.6571, 0.5567, 0.4648, 0.4201, 0.3939, 0.3684, 0.3434, 0.3157, 0.2933, 0.2647, 0.2408, 0.2184, 0.1960, 0.1825, 0.1668, 0.1594, 0.1473, 0.1410, 0.1386, 0.1265, 0.1305, 0.1213, 0.1279, 0.1170],
            'val_losses': [0.6832, 0.6174, 0.5056, 0.4496, 0.4260, 0.4060, 0.3955, 0.3759, 0.3630, 0.3436, 0.3298, 0.3220, 0.3075, 0.3174, 0.3034, 0.3090, 0.3295, 0.3334, 0.3541, 0.3552, 0.3617, 0.3713, 0.4004, 0.4146, 0.4043],
            'train_metrics': {
                'accuracy': 0.8167,
                'precision': 0.8186,
                'recall': 0.9073,
                'f1_score': 0.8607,
                'roc_auc': 0.9520,
                'average_precision': 0.9184
            },
            'val_metrics': {
                'accuracy': 0.8100,
                'precision': 0.8120,
                'recall': 0.9000,
                'f1_score': 0.8530,
                'roc_auc': 0.9450,
                'average_precision': 0.9100
            },
            'test_metrics': {
                'accuracy': 0.8531,
                'precision': 0.8186,
                'recall': 0.9073,
                'f1_score': 0.8607,
                'roc_auc': 0.9520,
                'average_precision': 0.9184
            }
        },
        'ComplEx': {
            'train_losses': [0.6840, 0.6082, 0.4806, 0.4162, 0.3750, 0.3353, 0.2935, 0.2543, 0.2184, 0.1822, 0.1656, 0.1506, 0.1328, 0.1317, 0.1291, 0.1181, 0.1219, 0.1178, 0.1110, 0.1171],
            'val_losses': [0.6642, 0.5412, 0.4523, 0.4153, 0.3941, 0.3621, 0.3327, 0.3134, 0.2949, 0.2826, 0.3007, 0.3263, 0.3259, 0.3514, 0.3620, 0.4043, 0.3959, 0.4022, 0.4211, 0.4287],
            'train_metrics': {
                'accuracy': 0.8113,
                'precision': 0.8445,
                'recall': 0.7630,
                'f1_score': 0.8017,
                'roc_auc': 0.9342,
                'average_precision': 0.9966
            },
            'val_metrics': {
                'accuracy': 0.8050,
                'precision': 0.8380,
                'recall': 0.7570,
                'f1_score': 0.7950,
                'roc_auc': 0.9280,
                'average_precision': 0.9900
            },
            'test_metrics': {
                'accuracy': 0.8113,
                'precision': 0.8445,
                'recall': 0.7630,
                'f1_score': 0.8017,
                'roc_auc': 0.9342,
                'average_precision': 0.9966
            }
        },
        'GCN': {
            'train_losses': [0.5949, 0.2000, 0.1630, 0.1417, 0.1174, 0.1087, 0.0883, 0.0713, 0.0723, 0.0685, 0.0656, 0.0675, 0.0588],
            'val_losses': [0.3073, 0.1897, 0.1616, 0.1913, 0.2232, 0.3038, 0.4067, 0.4852, 0.4529, 0.5127, 0.4910, 0.5530, 0.6427],
            'train_metrics': {
                'accuracy': 0.8728,
                'precision': 0.9770,
                'recall': 0.7635,
                'f1_score': 0.8571,
                'roc_auc': 0.9690,
                'average_precision': 0.9962
            },
            'val_metrics': {
                'accuracy': 0.8650,
                'precision': 0.9700,
                'recall': 0.7570,
                'f1_score': 0.8500,
                'roc_auc': 0.9620,
                'average_precision': 0.9900
            },
            'test_metrics': {
                'accuracy': 0.8728,
                'precision': 0.9770,
                'recall': 0.7635,
                'f1_score': 0.8571,
                'roc_auc': 0.9690,
                'average_precision': 0.9962
            }
        },
        'GraphSAGE': {
            'train_losses': [0.4693, 0.2315, 0.2224, 0.2087, 0.2040, 0.1912, 0.1793, 0.1555, 0.1335, 0.1143, 0.0995],
            'val_losses': [0.2735, 0.2778, 0.3520, 0.6512, 0.9650, 1.3167, 1.4430, 1.7747, 1.4945, 1.8557, 1.3601],
            'train_metrics': {
                'accuracy': 0.8103,
                'precision': 0.8419,
                'recall': 0.7640,
                'f1_score': 0.8010,
                'roc_auc': 0.8839,
                'average_precision': 0.9947
            },
            'val_metrics': {
                'accuracy': 0.8050,
                'precision': 0.8360,
                'recall': 0.7580,
                'f1_score': 0.7950,
                'roc_auc': 0.8780,
                'average_precision': 0.9900
            },
            'test_metrics': {
                'accuracy': 0.8103,
                'precision': 0.8419,
                'recall': 0.7640,
                'f1_score': 0.8010,
                'roc_auc': 0.8839,
                'average_precision': 0.9947
            }
        }
    }
    print(f"✅ Mock results created for {len(results)} models")

def plot_training_curves_dynamic(models_results):
    """
    Plot training curves for all models with dynamic subplot layout
    """
    
    print("Plotting training curves...")
    
    num_models = len(models_results)
    
    # Calculate subplot layout
    if num_models == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif num_models == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    elif num_models == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    elif num_models == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    else:
        # For more than 4 models, use a grid
        rows = (num_models + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
        axes = axes.flatten()
    
    fig.suptitle('Training Curves for Graph Embedding Models', fontsize=16, fontweight='bold')
    
    models = list(models_results.keys())
    
    for i, model_name in enumerate(models):
        train_losses = models_results[model_name]['train_losses']
        val_losses = models_results[model_name]['val_losses']
        
        axes[i].plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        axes[i].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
        axes[i].set_title(f'{model_name} Training Curves', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        
        # Add final loss values as text
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        axes[i].text(0.02, 0.98, f'Final Train: {final_train_loss:.4f}\nFinal Val: {final_val_loss:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_performance_comparison_dynamic(models_results):
    """
    Plot performance comparison across models with dynamic layout
    """
    
    print("Plotting performance comparison...")
    
    # Extract metrics
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
    
    num_metrics = len(metrics_names)
    num_models = len(models_results)
    
    # Calculate subplot layout
    if num_metrics <= 3:
        fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 8))
        if num_metrics == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    
    fig.suptitle('Performance Comparison Across Models', fontsize=16, fontweight='bold')
    
    models = list(models_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, num_models))
    
    for i, metric in enumerate(metrics_names):
        values = [models_results[model]['test_metrics'][metric] for model in models]
        
        bars = axes[i].bar(models, values, color=colors)
        axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(alpha=0.3, axis='y')
    
    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_metrics_heatmap(models_results):
    """
    Create a heatmap of all metrics for all models
    """
    
    print("Creating metrics heatmap...")
    
    # Prepare data for heatmap
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
    models = list(models_results.keys())
    
    # Create matrix
    data_matrix = []
    for model in models:
        row = [models_results[model]['test_metrics'][metric] for metric in metrics_names]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(metrics_names)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
    ax.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics_names)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', rotation=270, labelpad=20)
    
    ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_model_ranking(models_results):
    """
    Create a ranking plot showing best performing models for each metric
    """
    
    print("Creating model ranking plot...")
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
    models = list(models_results.keys())
    
    # Calculate rankings for each metric
    rankings = {}
    for metric in metrics_names:
        metric_values = [(model, models_results[model]['test_metrics'][metric]) for model in models]
        metric_values.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        rankings[metric] = metric_values
    
    # Create ranking plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create ranking matrix
    ranking_matrix = np.zeros((len(models), len(metrics_names)))
    for j, metric in enumerate(metrics_names):
        for i, (model, score) in enumerate(rankings[metric]):
            model_idx = models.index(model)
            ranking_matrix[model_idx, j] = i + 1  # Rank 1, 2, 3...
    
    # Create heatmap
    im = ax.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(models))
    
    # Set ticks and labels
    ax.set_xticks(range(len(metrics_names)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
    ax.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics_names)):
            rank = int(ranking_matrix[i, j])
            text = ax.text(j, i, f'{rank}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ranking (1=Best)', rotation=270, labelpad=20)
    
    ax.set_title('Model Rankings by Metric (1=Best Performance)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def create_performance_summary(models_results):
    """
    Create a comprehensive performance summary table matching the training output format
    """
    
    print("Creating performance summary...")
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
    models = list(models_results.keys())
    
    # Create summary DataFrame
    summary_data = []
    for model in models:
        row = {'Model': model}
        for metric in metrics_names:
            row[metric.replace('_', ' ').title()] = models_results[model]['test_metrics'][metric]
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display summary in the same format as training output
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Create table header
    print(f"{'Model':<12} {'Split':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    print("-" * 70)
    
    # Display results for each model (using test set performance)
    for model in models:
        metrics = models_results[model]['test_metrics']
        print(f"{model:<12} {'Test':<8} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f} {metrics['roc_auc']:<8.4f}")
    
    print("-" * 70)
    
    # Find best performing model for each metric
    print("\nBest Test Set Performance:")
    best_accuracy = max(models, key=lambda x: models_results[x]['test_metrics']['accuracy'])
    best_f1 = max(models, key=lambda x: models_results[x]['test_metrics']['f1_score'])
    best_auc = max(models, key=lambda x: models_results[x]['test_metrics']['roc_auc'])
    
    print(f"accuracy: {best_accuracy} ({models_results[best_accuracy]['test_metrics']['accuracy']:.4f})")
    print(f"f1_score: {best_f1} ({models_results[best_f1]['test_metrics']['f1_score']:.4f})")
    print(f"roc_auc: {best_auc} ({models_results[best_auc]['test_metrics']['roc_auc']:.4f})")
    
    # Calculate overall ranking
    print("\n" + "="*50)
    print("OVERALL MODEL RANKING")
    print("="*50)
    
    model_scores = {}
    for model in models:
        # Average of all metrics
        avg_score = np.mean([models_results[model]['test_metrics'][metric] for metric in metrics_names])
        model_scores[model] = avg_score
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(sorted_models, 1):
        print(f"{i}. {model}: {score:.4f}")
    
    return summary_df

def plot_loss_convergence(models_results):
    """
    Plot loss convergence analysis
    """
    
    print("Plotting loss convergence analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Loss Convergence Analysis', fontsize=16, fontweight='bold')
    
    models = list(models_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Plot 1: Training Loss Convergence
    ax1 = axes[0]
    for i, model in enumerate(models):
        train_losses = models_results[model]['train_losses']
        ax1.plot(train_losses, label=f'{model} Training', color=colors[i], linewidth=2)
    ax1.set_title('Training Loss Convergence')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Validation Loss Convergence
    ax2 = axes[1]
    for i, model in enumerate(models):
        val_losses = models_results[model]['val_losses']
        ax2.plot(val_losses, label=f'{model} Validation', color=colors[i], linewidth=2)
    ax2.set_title('Validation Loss Convergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_comprehensive_comparison(models_results):
    """
    Create a comprehensive comparison plot showing all models side by side
    """
    
    print("Creating comprehensive model comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Graph Embedding Models Comparison', fontsize=20, fontweight='bold')
    
    models = list(models_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Plot 1: Training Loss Curves
    ax1 = axes[0, 0]
    for i, model in enumerate(models):
        train_losses = models_results[model]['train_losses']
        ax1.plot(train_losses, label=f'{model}', color=colors[i], linewidth=2, marker='o', markersize=3)
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Validation Loss Curves
    ax2 = axes[0, 1]
    for i, model in enumerate(models):
        val_losses = models_results[model]['val_losses']
        ax2.plot(val_losses, label=f'{model}', color=colors[i], linewidth=2, marker='s', markersize=3)
    ax2.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Accuracy Comparison
    ax3 = axes[0, 2]
    accuracies = [models_results[model]['test_metrics']['accuracy'] for model in models]
    bars = ax3.bar(models, accuracies, color=colors)
    ax3.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: F1 Score Comparison
    ax4 = axes[1, 0]
    f1_scores = [models_results[model]['test_metrics']['f1_score'] for model in models]
    bars = ax4.bar(models, f1_scores, color=colors)
    ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('F1 Score')
    ax4.set_ylim(0, 1)
    for bar, f1 in zip(bars, f1_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(alpha=0.3, axis='y')
    
    # Plot 5: ROC-AUC Comparison
    ax5 = axes[1, 1]
    roc_aucs = [models_results[model]['test_metrics']['roc_auc'] for model in models]
    bars = ax5.bar(models, roc_aucs, color=colors)
    ax5.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylabel('ROC-AUC')
    ax5.set_ylim(0, 1)
    for bar, auc in zip(bars, roc_aucs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(alpha=0.3, axis='y')
    
    # Plot 6: Overall Performance Comparison (Bar Chart instead of Radar)
    ax6 = axes[1, 2]
    
    # Calculate overall performance score for each model
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    overall_scores = []
    
    for model in models:
        # Average of all metrics
        avg_score = np.mean([models_results[model]['test_metrics'][metric] for metric in metrics])
        overall_scores.append(avg_score)
    
    bars = ax6.bar(models, overall_scores, color=colors)
    ax6.set_title('Overall Performance Score', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Average Score')
    ax6.set_ylim(0, 1)
    
    for bar, score in zip(bars, overall_scores):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_radar_chart(models_results):
    """
    Create a proper radar chart for model comparison
    """
    
    print("Creating radar chart comparison...")
    
    # Create a separate figure for radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    models = list(models_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Prepare data for radar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    for i, model in enumerate(models):
        values = [models_results[model]['test_metrics'][metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Execute all visualizations
if 'results' in locals() and results:
    print("Creating comprehensive visualizations for all trained models...")
    print(f"Models to visualize: {list(results.keys())}")
    
    # 1. Comprehensive comparison (main visualization)
    plot_comprehensive_comparison(results)
    
    # 2. Training curves
    plot_training_curves_dynamic(results)
    
    # 3. Performance comparison
    plot_performance_comparison_dynamic(results)
    
    # 4. Metrics heatmap
    plot_metrics_heatmap(results)
    
    # 5. Model ranking
    plot_model_ranking(results)
    
    # 6. Loss convergence
    plot_loss_convergence(results)
    
    # 7. Performance summary (matching training output format)
    summary_df = create_performance_summary(results)
    
    # 8. Optional: Radar chart (uncomment if you want to see it)
    # plot_radar_chart(results)
    
    print("\n🎉 All visualizations completed!")
    print("="*60)
    print("📊 Visualization Summary:")
    print("   ✅ Comprehensive model comparison (6-panel overview)")
    print("   ✅ Training curves with dynamic layout")
    print("   ✅ Performance comparison bar charts")
    print("   ✅ Metrics heatmap")
    print("   ✅ Model ranking analysis")
    print("   ✅ Loss convergence plots")
    print("   ✅ Performance summary (training output format)")
    print("   ✅ Optional radar chart (uncomment to enable)")
    
else:
    print("❌ No results found. Please run the graph embeddings training cell first.")
