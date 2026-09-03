from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

print(data.feature_names) 
print(f"Nome das Classes: {data.target_names}")
print(f"Número de Atributos: {len(data.feature_names)}")
print(f"Exemplo de linha de dados: {X[0][:5]}...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acuracia_knn = accuracy_score(y_test, y_pred_knn)

arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)               
y_pred_arvore = arvore.predict(X_test)
acuracia_arvore = accuracy_score(y_test, y_pred_arvore)

print(f"\nAcurácia KNN: {acuracia_knn * 100:.2f}%")
print(f"Acurácia Árvore de Decisão: {acuracia_arvore * 100:.2f}%")