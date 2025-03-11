from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load the trained model
model = AegisVisionModel(input_dim=1, hidden_dim=64, kernel_size=(3, 3), num_layers=3).to(device)
checkpoint = torch.load("aegis_vision_checkpoint_epoch_10.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = np.expand_dims(frame, axis=0)  # Add channel dimension
    return torch.from_numpy(frame)

# Detect anomalies in a frame
def detect_anomaly(frame):
    with torch.no_grad():
        frame_tensor = preprocess_frame(frame).to(device)
        output = model(frame_tensor)
        anomaly_score = torch.mean((output - frame_tensor) ** 2).item()
        return anomaly_score

# Flask route to process video frames
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    frame_data = data['frame']
    
    # Decode base64 frame
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    
    # Detect anomaly
    anomaly_score = detect_anomaly(frame)
    
    # Mark anomaly on the frame
    if anomaly_score > 0.5:  # Threshold for anomaly detection
        cv2.putText(frame, "Anomaly Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Encode frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'frame': frame_encoded,
        'anomaly_score': anomaly_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)