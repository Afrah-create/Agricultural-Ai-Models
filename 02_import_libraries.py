"""
Phase 1, Cell 2: Import Libraries and Basic Configuration
This cell imports all necessary libraries and sets up basic configuration
"""

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Google Generative AI
import google.generativeai as genai

# PDF processing
import PyPDF2
import pdfplumber
from io import BytesIO

# Graph processing
import networkx as nx
from pyvis.network import Network

# Machine Learning
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphSAGE
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Knowledge Graph
from rdflib import Graph, Namespace, Literal, URIRef
import owlready2

# Optimization
from ortools.linear_solver import pywraplp

# Utilities
import os
import json
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("All libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NetworkX version: {nx.__version__}")
print(f"Transformers version: {transformers.__version__}")
