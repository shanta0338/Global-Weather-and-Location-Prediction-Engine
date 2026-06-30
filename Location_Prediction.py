import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

TARGET_LOCATION = "location_name"
TARGET_COUNTRY  = "country"

target_col = TARGET_COUNTRY 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv(r"D:\My Folder\Dataset\GlobalWeatherRepository.csv")

features = [
    'temperature_celsius', 'wind_kph', 'wind_degree', 'pressure_mb', 
    'precip_mm', 'humidity', 'cloud', 'visibility_km', 'uv_index', 'gust_kph'
]

# Clean data by removing rows with missing values in our features or target
df = df.dropna(subset=features + [target_col])

X = df[features].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_col])
num_classes = len(label_encoder.classes_)
print(f"Number of unique classes to predict: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# Standardise the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class WeatherClassificationDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        # Note: Targets for CrossEntropyLoss must be of type torch.long
        self.y = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = WeatherClassificationDataset(X_train_scaled, y_train)
test_dataset = WeatherClassificationDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class LocationClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LocationClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Output matches number of classes
        )
        
    def forward(self, x):
        return self.network(x)

# Instantiate the model and move it to the device immediately
input_dimension = len(features)
model = LocationClassifier(input_dim=input_dimension, output_dim=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimiser = optim.NAdam(model.parameters(), lr=0.01)

epochs = 5

print("Beginning training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_X, batch_y in train_loader:
        
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimiser.zero_grad()
        
        # Forward pass yields unnormalised logits
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Calculate accuracy for this batch
        _, predicted = torch.max(outputs, 1)
        total_samples += batch_y.size(0)
        correct_predictions += (predicted == batch_y).sum().item()
        
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_samples
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

