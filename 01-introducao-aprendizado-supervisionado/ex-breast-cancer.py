from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dados = load_breast_cancer(as_frame=True)
X = dados.data
y = dados.target

print ("--- VISÃO GERAL DOS DADOS ---")
print (f"Formato de X (linhas, colunas/características): {X.shape}")
print (f"Formato de y (0 = Maligno, 1 = Benigno): {dados.target_names}")
print (f"\nPrimeiras linhas dos atributos: ")
print (X.head())
