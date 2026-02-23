import torch 
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import dagshub
import mlflow
import mlflow.pytorch
import torch.optim as optim

##########################################
# Dagshub and Mlflow setup
##########################################

dagshub.init(
    repo_owner="patelyash9404",
    repo_name="ANN-implementation-with-MLflow-and-CI",
    mlflow=True
)

mlflow.set_experiment("ANN-Baseline")

###########################################
# Load the dataset
###########################################

df = pd.read_csv(r"data/eda.csv")

#############################################
# Train and test split
#############################################
df['class'] = df['class'] - 1
X = df.drop(['class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#############################################
# Scale the data 
#############################################

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

##############################################
# Model define
##############################################
class ANNModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, features):
        return self.network(features)

#############################################
# Hyperparameters
#############################################
params = {
    'learning_rate': [0.02, 0.01, 0.001],
    'epochs': [5, 10, 15]
}

loss_function = nn.CrossEntropyLoss()

############################################
# Mlflow experimentation
############################################

best_acc = 0

for lr in params['learning_rate']:
    for ep in params["epochs"]:

        with mlflow.start_run():
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", ep)

            model = ANNModel(
                num_features=X_train.shape[1],
                num_classes=len(y.unique())
            )

            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Training loop
            for epoch in range(ep):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = loss_function(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                preds = model(X_test)
                y_pred = torch.argmax(preds, dim=1).numpy()
                acc = accuracy_score(y_test.numpy(), y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("loss", loss.item())

            # Log model
            mlflow.pytorch.log_model(model, "model")

            print(f"LR={lr}, Epochs={ep}, Accuracy={acc:.4f}")

            if acc > best_acc:
                best_acc = acc

print("Best Accuracy:", best_acc)

