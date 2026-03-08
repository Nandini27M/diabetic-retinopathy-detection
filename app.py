from flask import Flask, request, jsonify, send_from_directory
from model import HybridModel
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# --- Model Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(num_classes=5)
model.load_state_dict(torch.load('best_model_amp.pth', map_location=device))
model.to(device)
model.eval()

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Class Names (Matches Frontend) ---
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

@app.route('/')
def index():
    """Serves the frontend HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
            pred_class = int(probs.argmax())
            pred_class_name = CLASS_NAMES[pred_class]

        return jsonify({
            'predicted_class': pred_class,
            'predicted_class_name': pred_class_name,
            'class_probabilities': [float(x) for x in probs]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
