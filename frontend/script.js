const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const anomalyScoreElement = document.getElementById('anomaly-score');

// Start video stream from the user's camera
async function startVideo() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (error) {
    console.error('Error accessing camera:', error);
  }
}

// Send frames to the backend for processing
async function processFrame() {
  if (!video || !canvas) return;

  // Draw video frame on canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert canvas image to base64
  const frame = canvas.toDataURL('image/jpeg');

  // Send frame to the backend
  try {
    const response = await fetch('http://localhost:5000/process_frame', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ frame }),
    });

    const data = await response.json();

    // Display processed frame
    const img = new Image();
    img.src = `data:image/jpeg;base64,${data.frame}`;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    // Update anomaly score
    anomalyScoreElement.textContent = data.anomaly_score.toFixed(4);
  } catch (error) {
    console.error('Error processing frame:', error);
  }
}

// Start video and process frames
startVideo();
setInterval(processFrame, 1000);  // Process 1 frame per second