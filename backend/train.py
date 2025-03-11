import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 4
sequence_length = 10
num_epochs = 50
learning_rate = 0.001
input_size = (256, 256)  # Resize frames to this size
hidden_dim = 64
kernel_size = (3, 3)
num_layers = 3

# ConvLSTM Cell implementation
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# ConvLSTM Model
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim, self.kernel_size))
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        # Input shape: (batch, sequence, channel, height, width)
        # Output shape: (batch, sequence, hidden_dim, height, width)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []
        
        # Unroll over time steps
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                if layer_idx == 0:
                    x = input_tensor[:, t, :, :, :]
                else:
                    x = layer_output_list[layer_idx-1][:, t, :, :, :]
                
                h, c = self.cell_list[layer_idx](x, (h, c))
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
            
        return layer_output_list[-1], last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

# Autoencoder model with ConvLSTM
class AegisVisionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=(3, 3), num_layers=3):
        super(AegisVisionModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(16, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.convlstm = ConvLSTM(input_dim=hidden_dim, 
                                 hidden_dim=hidden_dim, 
                                 kernel_size=kernel_size, 
                                 num_layers=num_layers,
                                 batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 128x128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 256x256
            nn.Conv2d(32, input_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Encode each frame
        encoded_frames = []
        for t in range(seq_len):
            encoded_frame = self.encoder(x[:, t])
            encoded_frames.append(encoded_frame)
        
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        # Process with ConvLSTM
        output, _ = self.convlstm(encoded_sequence)
        
        # Decode each output frame
        decoded_frames = []
        for t in range(seq_len):
            decoded_frame = self.decoder(output[:, t])
            decoded_frames.append(decoded_frame)
        
        decoded_sequence = torch.stack(decoded_frames, dim=1)
        
        return decoded_sequence

# Custom Dataset for CUHK Avenue - FIXED VERSION
class AvenueDataset(Dataset):
    def __init__(self, root_dir, is_train=True, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Define paths for videos
        folder_type = "training" if is_train else "testing"
        self.video_dir = os.path.join(root_dir, f"{folder_type}_videos")
        
        # Get list of all video files
        self.video_paths = [os.path.join(self.video_dir, f) for f in sorted(os.listdir(self.video_dir)) if f.endswith('.avi')]
        
        # Create list of all frame sequences
        self.frame_sequences = []
        for video_idx, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Generate sequences with overlapping frames
            for i in range(0, frame_count - self.sequence_length + 1, 1):
                self.frame_sequences.append((video_idx, video_path, i))
                
    def __len__(self):
        return len(self.frame_sequences)
    
    def __getitem__(self, idx):
        video_idx, video_path, start_frame = self.frame_sequences[idx]
        
        # Pre-allocate numpy array for frames
        frames_array = np.zeros((self.sequence_length, 1, input_size[0], input_size[1]), dtype=np.float32)
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, input_size)
            
            # Normalize and add to array
            frames_array[i, 0] = frame.astype(np.float32) / 255.0
            
        cap.release()
        
        # Convert numpy array to tensor directly
        frames_tensor = torch.from_numpy(frames_array)
        
        return frames_tensor, video_idx, start_frame

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (batch_frames, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move batch to device
            batch_frames = batch_frames.to(device)
            
            # Forward pass
            outputs = model(batch_frames)
            
            # Calculate reconstruction loss
            loss = criterion(outputs, batch_frames)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
        # Save model checkpoint
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f"aegis_vision_checkpoint_epoch_{epoch+1}.pth")
    
    return train_losses

# Test function to detect anomalies
def test_model(model, test_loader, device):
    model.eval()
    anomaly_scores = []
    frame_info = []  # To store (video_index, frame_number) for each prediction
    
    with torch.no_grad():
        for batch_frames, video_idxs, start_frames in tqdm(test_loader, desc="Evaluating"):
            batch_frames = batch_frames.to(device)
            
            # Forward pass
            outputs = model(batch_frames)
            
            # Calculate reconstruction error for each frame in the sequence
            for i in range(batch_frames.size(0)):
                for t in range(batch_frames.size(1)):
                    # MSE for each frame
                    mse = torch.mean((outputs[i, t] - batch_frames[i, t]) ** 2).item()
                    anomaly_scores.append(mse)
                    
                    # Store frame info
                    frame_info.append((video_idxs[i].item(), start_frames[i].item() + t))
    
    return anomaly_scores, frame_info

# Function to load ground truth anomaly labels
def load_ground_truth(dataset_path):
    ground_truth_dir = os.path.join(dataset_path, "ground_truth_demo/testing_label_mask")
    ground_truth_files = sorted([os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir) if f.endswith('.mat')])
    
    all_labels = []
    for gt_file in ground_truth_files:
        # Load .mat file
        mat_data = sio.loadmat(gt_file)
        label = mat_data['volLabel']
        all_labels.extend([1 if np.sum(frame) > 0 else 0 for frame in label])
    
    return np.array(all_labels)

# Evaluate anomaly detection performance
def evaluate_performance(anomaly_scores, ground_truth):
    # Normalize anomaly scores
    anomaly_scores = np.array(anomaly_scores)
    anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(ground_truth, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return roc_auc

# Visualize results
def visualize_anomalies(dataset_path, frame_info, anomaly_scores, threshold=0.8):
    # Get top anomalies
    anomaly_scores = np.array(anomaly_scores)
    normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    
    top_indices = np.where(normalized_scores > threshold)[0]
    
    # Create output directory
    os.makedirs("anomaly_frames", exist_ok=True)
    
    # Get mapping of video indices to video paths
    video_dir = os.path.join(dataset_path, "testing_videos")
    video_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.avi')])
    
    # Extract and save top anomaly frames
    saved_frames = 0
    for idx in top_indices[:50]:  # Save top 50 anomaly frames
        video_idx, frame_num = frame_info[idx]
        video_path = video_paths[video_idx]
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            save_path = f"anomaly_frames/video_{video_idx}_frame_{frame_num}_score_{normalized_scores[idx]:.4f}.jpg"
            cv2.imwrite(save_path, frame)
            saved_frames += 1
    
    print(f"Saved {saved_frames} anomaly frames to 'anomaly_frames' directory")

# Main function
def main():
    # Set dataset path (REPLACE THIS WITH YOUR ACTUAL PATH)
    dataset_path = ""  # <-- REPLACE THIS PATH
    
    # Initialize model
    model = AegisVisionModel(input_dim=1, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Create datasets and dataloaders
    train_dataset = AvenueDataset(root_dir=dataset_path, is_train=True, sequence_length=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    test_dataset = AvenueDataset(root_dir=dataset_path, is_train=False, sequence_length=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training with {len(train_dataset)} sequences")
    print(f"Testing with {len(test_dataset)} sequences")
    
    # Train the model
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "aegis_vision_final_model.pth")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Test the model and calculate anomaly scores
    print("Testing model...")
    anomaly_scores, frame_info = test_model(model, test_loader, device)
    
    # Load ground truth
    ground_truth = load_ground_truth(dataset_path)
    
    # Evaluate performance
    roc_auc = evaluate_performance(anomaly_scores, ground_truth)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save results
    np.save('anomaly_scores.npy', np.array(anomaly_scores))
    
    # Visualize anomalies
    visualize_anomalies(dataset_path, frame_info, anomaly_scores)

if __name__ == "__main__":
    main()