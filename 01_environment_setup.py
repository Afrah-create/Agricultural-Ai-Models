"""
Phase 1, Cell 1: Environment Setup and Package Installation
This cell sets up the Google Colab environment with all required packages
"""

!pip install google-generativeai
!pip install PyPDF2 pdfplumber
!pip install pandas numpy matplotlib seaborn plotly
!pip install networkx pyvis
!pip install torch torch-geometric
!pip install transformers datasets
!pip install scikit-learn scipy
!pip install rdflib owlready2
!pip install ortools
!pip install google-cloud-storage
!pip install requests beautifulsoup4
!pip install folium
!pip install openpyxl xlsxwriter

print("All packages installed successfully!")
print("Package versions:")
import pkg_resources
packages = ['google-generativeai', 'pandas', 'numpy', 'torch', 'networkx', 'transformers']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"  {pkg}: {version}")
    except:
        print(f"  {pkg}: Not found")
