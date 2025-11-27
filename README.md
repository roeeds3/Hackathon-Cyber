# Cyber Attack Classifier for EV Charging Stations

A machine learning-powered API system for detecting and analyzing cyber attacks on Electric Vehicle (EV) charging stations. This system uses a Random Forest Classifier to identify attacks in real-time and performs spatial cluster analysis to detect coordinated attack patterns.

## 🎯 Overview

This project provides a comprehensive solution for monitoring EV charging station networks and detecting potential cyber attacks. It combines:

- **ML-based Attack Detection**: Uses a trained Random Forest Classifier to classify charging station sensor data
- **Spatial Cluster Analysis**: Identifies coordinated attacks using HDBSCAN clustering
- **Risk Assessment**: Calculates severity scores and identifies at-risk nodes
- **Real-time Visualization**: Generates visual representations of attack clusters and vulnerable areas

## ✨ Features

- **Single & Batch Classification**: Process individual or multiple charging stations simultaneously
- **Attack Severity Scoring**: Computes 0-100 severity scores based on sensor anomalies and ML probabilities
- **Cluster Detection**: Identifies coordinated attacks using spatial clustering (HDBSCAN)
- **At-Risk Node Detection**: Finds safe nodes that may be vulnerable due to proximity to attack clusters
- **Interactive Visualization**: Generate PNG visualizations showing attack patterns, clusters, and severity levels
- **RESTful API**: Clean FastAPI-based endpoints with automatic documentation

## 🏗️ Architecture

### Core Components

1. **`main.py`**: FastAPI application entry point and server configuration
2. **`api_router.py`**: API endpoints and request/response models
3. **`classify_one.py`**: ML model loading, preprocessing, and prediction logic
4. **`Predictor.py`**: Cluster detection, visualization, and streaming attack monitoring

### System Flow

```
Sensor Data → Preprocessing → ML Classification → Severity Calculation
                                                      ↓
                                            Cluster Detection (HDBSCAN)
                                                      ↓
                                            At-Risk Node Identification
                                                      ↓
                                            Visualization Generation
```

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

### Key Dependencies

- `fastapi`: Web framework
- `scikit-learn`: Machine learning model
- `hdbscan`: Clustering algorithm
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `joblib`: Model serialization

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Hackathon-Cyber
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files exist**
   - `classifier model/cyber_rfc_model.joblib` (ML model)
   - `classifier model/cyber_rfc_meta.json` (Model metadata)

## 💻 Usage

### Starting the Server

#### Option 1: Using the startup script
```bash
chmod +x start_server.sh
./start_server.sh
```

This script will:
- Start the FastAPI server on port 8000
- Create an ngrok tunnel for public access (if configured)

#### Option 2: Manual start
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Root endpoint**: `http://localhost:8000/`

## 📡 API Endpoints

### `GET /`
Root endpoint providing API information and available endpoints.

**Response:**
```json
{
  "message": "Cyber Attack Classifier API",
  "version": "1.0.0",
  "endpoints": {
    "classify": "/classify - POST endpoint for classifying a single charging station data",
    "classify-batch": "/classify-batch - POST endpoint for classifying multiple charging station data in batch",
    "visualize": "/visualize - GET endpoint for getting attack cluster visualization (returns PNG image)"
  }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `POST /classify`
Classify a single charging station for cyber attacks.

**Request Body:**
```json
{
  "charger_id": 123,
  "current": 15.2,
  "delta_current": 0.5,
  "voltage": 480.1,
  "delta_voltage": 1.1,
  "power_w": 7296.5,
  "expected_load": 7300,
  "status_str": "CHARGING",
  "loc_x": 40.71,
  "loc_y": -74.01,
  "temperature": 35.5,
  "provider": "EVGO"
}
```

**Response:**
```json
{
  "ID": "123",
  "Is_attacked": false,
  "loc_x": 40.71,
  "loc_y": -74.01,
  "provider": "EVGO",
  "severity": null
}
```

**Note:** `severity` is only provided when `Is_attacked` is `true` (range: 0-100).

### `POST /classify-batch`
Classify multiple charging stations in a single request.

**Request Body:**
```json
{
  "chargers": [
    {
      "charger_id": 123,
      "current": 15.2,
      "delta_current": 0.5,
      "voltage": 480.1,
      "delta_voltage": 1.1,
      "power_w": 7296.5,
      "expected_load": 7300,
      "status_str": "CHARGING",
      "loc_x": 40.71,
      "loc_y": -74.01,
      "temperature": 35.5,
      "provider": "EVGO"
    },
    {
      "charger_id": 124,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "total_processed": 2,
  "total_attacked": 1,
  "results": [
    {
      "ID": "123",
      "Is_attacked": false,
      "loc_x": 40.71,
      "loc_y": -74.01,
      "provider": "EVGO",
      "severity": null
    },
    {
      "ID": "124",
      "Is_attacked": true,
      "loc_x": 50.0,
      "loc_y": -80.0,
      "provider": "Tesla",
      "severity": 75.5
    }
  ]
}
```

### `GET /visualize`
Get a PNG visualization of attack clusters and at-risk nodes.

**Response:** PNG image (Content-Type: `image/png`)

The visualization shows:
- **Gray circles**: Safe charging stations
- **Orange circles**: At-risk nodes (safe but near attack clusters)
- **Colored circles**: Attacked nodes grouped by cluster
- **Red circles**: Isolated attacks (noise points)
- **Circles around clusters**: Cluster boundaries
- **Severity labels**: Numeric severity scores displayed on attacked nodes

## 🔬 Model Details

### Classification Model

- **Algorithm**: Random Forest Classifier
- **Classes**: 
  - `0`: OK (Normal operation)
  - `1`: ERROR (System error, not an attack)
  - `2`: CYBER_ATTACK (Detected attack)

### Input Features

The model uses the following features (after dropping charger ID and location):
- `current`: Current reading (A)
- `delta_current`: Change in current (A)
- `voltage`: Voltage reading (V)
- `delta_voltage`: Change in voltage (V)
- `power (W)`: Power consumption (W)
- `expected_load`: Expected load (W)
- `status`: Encoded status (CHARGING, IDLE, OFF, etc.)
- `temperature`: Temperature reading (°C)

### Severity Calculation

Severity scores (0-100) are computed using a hybrid approach:

1. **ML Probability**: Attack probability from the classifier
2. **Rule-based Factors**:
   - Power mismatch (calculated vs. measured)
   - Load mismatch (expected vs. actual)
   - Current/voltage delta spikes
   - Temperature anomalies
   - Status contradictions

Formula: `severity = 100 × p_attack + 20 × temp_score + 15 × power_mismatch`

## 🎨 Cluster Detection

### HDBSCAN Clustering

The system uses HDBSCAN (Hierarchical Density-Based Spatial Clustering) to identify coordinated attacks:

- **Features**: Location (x, y) and provider encoding
- **Min Cluster Size**: 3 nodes (configurable)
- **Distance Threshold**: 300 units (configurable)

### At-Risk Node Detection

Safe nodes are identified as "at-risk" if they would cluster together with attacked nodes when re-clustered with the full dataset. This helps identify potentially vulnerable stations before they're attacked.

## 📝 Example Usage

### Python Example

```python
import requests

# Single classification
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "charger_id": 123,
        "current": 15.2,
        "delta_current": 0.5,
        "voltage": 480.1,
        "delta_voltage": 1.1,
        "power_w": 7296.5,
        "expected_load": 7300,
        "status_str": "CHARGING",
        "loc_x": 40.71,
        "loc_y": -74.01,
        "temperature": 35.5,
        "provider": "EVGO"
    }
)
print(response.json())

# Get visualization
response = requests.get("http://localhost:8000/visualize")
with open("attack_clusters.png", "wb") as f:
    f.write(response.content)
```

### cURL Example

```bash
# Classify single charger
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "charger_id": 123,
    "current": 15.2,
    "delta_current": 0.5,
    "voltage": 480.1,
    "delta_voltage": 1.1,
    "power_w": 7296.5,
    "expected_load": 7300,
    "status_str": "CHARGING",
    "loc_x": 40.71,
    "loc_y": -74.01,
    "temperature": 35.5,
    "provider": "EVGO"
  }'

# Get visualization
curl -X GET "http://localhost:8000/visualize" \
  --output attack_clusters.png
```

## 🔧 Configuration

### Cluster Detection Parameters

In `api_router.py`, you can adjust:
- `distance_threshold`: Spatial distance threshold for clustering (default: 300)
- `min_cluster_size`: Minimum nodes required to form a cluster (default: 3)

```python
monitor = StreamingAttackMonitor(distance_threshold=300, min_cluster_size=3)
```

### Server Configuration

In `start_server.sh`:
- `FASTAPI_PORT`: Server port (default: 8000)
- `NGROK_DOMAIN`: ngrok domain for public access (optional)

## 📁 Project Structure

```
Hackathon-Cyber/
├── main.py                 # FastAPI application entry point
├── api_router.py           # API endpoints and routing
├── classify_one.py         # ML model loading and prediction
├── Predictor.py            # Cluster detection and visualization
├── start_server.sh         # Server startup script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
└── classifier model/      # ML model directory
    ├── cyber_rfc_model.joblib    # Trained Random Forest model
    ├── cyber_rfc_meta.json       # Model metadata
    ├── Cyber_Attack_Classifier_ML_Model_v2.ipynb  # Model training notebook
    └── EV_charging_simulated_data.csv             # Training data
```

## 🧪 Testing

After starting the server, you can test the endpoints using:

1. **Swagger UI**: Visit `http://localhost:8000/docs` for interactive API testing
2. **Health Check**: `curl http://localhost:8000/health`
3. **Root Endpoint**: `curl http://localhost:8000/`

## 📊 Status Values

The model supports the following status strings (defined in `cyber_rfc_meta.json`):
- `CHARGING`: Station is actively charging
- `IDLE`: Station is idle/available
- `OFF`: Station is turned off
- (Additional statuses may be defined in the metadata)

## ⚠️ Error Handling

The API handles errors gracefully:

- **400 Bad Request**: Invalid input data (e.g., unknown status string)
- **404 Not Found**: No data available for visualization
- **500 Internal Server Error**: Unexpected server errors

Errors in cluster detection don't fail classification requests - classification results are still returned even if visualization fails.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Note**: This system is designed for monitoring and detection purposes. Always verify attack detections through additional security measures and follow your organization's incident response procedures.
