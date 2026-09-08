import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetClassifier(nn.Module):
    def __init__(self):
        super(PointNetClassifier, self).__init__()

        # MLP layers for point feature extraction
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Fully connected layers for classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (0, 1)

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, pnum, 3]
        batch_size, pnum, _ = x.size()

        # Extract point features using MLP
        point_features = self.mlp1(x)  # Shape: [batch_size, pnum, 64]
        point_features = self.mlp2(point_features)  # Shape: [batch_size, pnum, 128]
        point_features = self.mlp3(point_features)  # Shape: [batch_size, pnum, 256]

        # Global feature aggregation using max pooling
        global_features = torch.max(point_features, dim=1, keepdim=True)[0]  # Shape: [batch_size, 1, 256]

        # Combine global features with local features
        # Broadcasting global features to match point features shape
        combined_features = point_features + global_features  # Shape: [batch_size, pnum, 256]

        # Classification for each point
        combined_features = F.relu(self.fc1(combined_features.view(-1, 256)))  # Flatten: [batch_size * pnum, 256]
        combined_features = self.fc2(combined_features)  # Shape: [batch_size * pnum, 2]

        # Reshape back to [batch_size, pnum, 2]
        output = combined_features.view(batch_size, pnum, 2)

        return output

# 示例使用
if __name__ == '__main__':
    model = PointNetClassifier()
    input_tensor = torch.randn(32, 4096, 3)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Output should be [32, 4096, 2]
