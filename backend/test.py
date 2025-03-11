import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 4
sequence_length = 10
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
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []
        
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
        batch_size, seq_len, c, h, w = x.size()
        
        encoded_frames = []
        for t in range(seq_len):
            encoded_frame = self.encoder(x[:, t])
            encoded_frames.append(encoded_frame)
        
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        output, _ = self.convlstm(encoded_sequence)
        
        decoded_frames = []
        for t in range(seq_len):
            decoded_frame = self.decoder(output[:, t])
            decoded_frames.append(decoded_frame)
        
        decoded_sequence = torch.stack(decoded_frames, dim=1)
        
        return decoded_sequence

# Custom Dataset for CUHK Avenue
class AvenueDataset(Dataset):
    def __init__(self, root_dir, is_train=True, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.sequence_length = sequence_length
        self.transform = transform
        
        folder_type = "training" if is_train else "testing"
        self.video_dir = os.path.join(root_dir, f"{folder_type}_videos")
        
        self.video_paths = [os.path.join(self.video_dir, f) for f in sorted(os.listdir(self.video_dir)) if f.endswith('.avi')]
        
        self.frame_sequences = []
        for video_idx, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            for i in range(0, frame_count - self.sequence_length + 1, 1):
                self.frame_sequences.append((video_idx, video_path, i))
                
    def __len__(self):
        return len(self.frame_sequences)
    
    def __getitem__(self, idx):
        video_idx, video_path, start_frame = self.frame_sequences[idx]
        
        frames_array = np.zeros((self.sequence_length, 1, input_size[0], input_size[1]), dtype=np.float32)
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, input_size)
            frames_array[i, 0] = frame.astype(np.float32) / 255.0
            
        cap.release()
        
        frames_tensor = torch.from_numpy(frames_array)
        
        return frames_tensor, video_idx, start_frame

# Function to load ground truth anomaly labels
def load_ground_truth(dataset_path):
    ground_truth_dir = os.path.join(dataset_path, "ground_truth_demo/testing_label_mask")
    ground_truth_files = sorted([os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir) if f.endswith('.mat')])
    
    all_labels = []
    for gt_file in ground_truth_files:
        # Load .mat file
        mat_data = sio.loadmat(gt_file)
        label = mat_data['volLabel']  # Assuming 'volLabel' is the key for ground truth labels
        # Flatten the labels and append to the list
        all_labels.extend(label.flatten())
    
    return np.array(all_labels)

# Function to visualize anomalies in videos
def visualize_anomalies(dataset_path, model, test_loader, ground_truth):
    # Get mapping of video indices to video paths
    video_dir = os.path.join(dataset_path, "testing_videos")
    video_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.avi')])
    
    # Create output directory
    os.makedirs("anomaly_videos", exist_ok=True)
    
    # Process each video
    for video_idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer
        output_path = f"anomaly_videos/video_{video_idx}_anomalies.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        for frame_num in tqdm(range(frame_count), desc=f"Processing video {video_idx}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if the frame is anomalous
            if ground_truth[frame_num].any():  # Use .any() to check if any element is True (1)
                # Draw rectangle around the anomaly
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 2)
                cv2.putText(frame, "Anomaly Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(frame)
        
        cap.release()
        out.release()
    
    print("Anomaly videos saved in 'anomaly_videos' directory")

# Main function
def main():
    dataset_path = ""
    
    model = AegisVisionModel(input_dim=1, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers).to(device)
    checkpoint = torch.load("aegis_vision_checkpoint_epoch_10.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_dataset = AvenueDataset(root_dir=dataset_path, is_train=False, sequence_length=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    ground_truth = load_ground_truth(dataset_path)
    
    visualize_anomalies(dataset_path, model, test_loader, ground_truth)

if __name__ == "__main__":
    main()