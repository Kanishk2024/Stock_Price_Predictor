#!/usr/bin/env python
# coding: utf-8

"""
Stock Price Predictor Backend API
Flask server that provides REST API endpoints for stock price predictions
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime
import os
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# LSTM Predictor Model
class Predictor(nn.Module):
    """LSTM-based stock price predictor"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Predictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def prepare_data(ticker):
    """
    Download and prepare stock data for prediction
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Prepared data dictionary with tensors and metadata
    """
    try:
        logger.info(f"Downloading data for {ticker}...")
        df = yf.download(ticker, '2020-01-01', progress=False)
        
        # Suppress the FutureWarning about auto_adjust
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Store original data for later use
        original_df = df.copy()
        
        # Normalize the closing price
        scaler = StandardScaler()
        df['Close'] = scaler.fit_transform(df[['Close']])
        
        # Create sequences
        seq_length = 30
        data = []
        
        for i in range(len(df) - seq_length):
            data.append(df['Close'].iloc[i:i+seq_length].values)
        
        data = np.array(data)
        logger.info(f"Data shape after stacking: {data.shape}")
        
        if len(data) < 50:
            raise ValueError("Not enough data to train the model")
        
        # Split data
        train_size = int(0.8 * len(data))
        
        # Reshape for LSTM: (batch_size, seq_length - 1, 1) for X and (batch_size, 1) for Y
        X_train = torch.from_numpy(data[:train_size, :-1].reshape(-1, seq_length - 1, 1)).type(torch.Tensor).to(device)
        Y_train = torch.from_numpy(data[:train_size, -1].reshape(-1, 1)).type(torch.Tensor).to(device)
        X_test = torch.from_numpy(data[train_size:, :-1].reshape(-1, seq_length - 1, 1)).type(torch.Tensor).to(device)
        Y_test = torch.from_numpy(data[train_size:, -1].reshape(-1, 1)).type(torch.Tensor).to(device)
        
        logger.info(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
        logger.info(f"Data prepared. Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'scaler': scaler,
            'df': original_df,
            'seq_length': seq_length,
            'train_size': train_size,
            'data': data
        }
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


def train_model(X_train, Y_train, num_epochs=200):
    """
    Train the LSTM model
    
    Args:
        X_train: Training input tensor
        Y_train: Training output tensor
        num_epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    logger.info("Training model...")
    
    model = Predictor(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)
        
        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    logger.info("Model training completed")
    return model


def make_predictions(model, data_dict):
    """
    Make predictions using the trained model
    
    Args:
        model: Trained LSTM model
        data_dict: Dictionary containing prepared data
        
    Returns:
        Dictionary with predictions and metrics
    """
    logger.info("Making predictions...")
    
    model.eval()
    with torch.no_grad():
        Y_train_pred = model(data_dict['X_train']).cpu().numpy()
        Y_test_pred = model(data_dict['X_test']).cpu().numpy()
    
    # Inverse transform to get actual prices
    Y_train_actual = data_dict['scaler'].inverse_transform(data_dict['Y_train'].cpu().numpy())
    Y_train_pred = data_dict['scaler'].inverse_transform(Y_train_pred)
    Y_test_actual = data_dict['scaler'].inverse_transform(data_dict['Y_test'].cpu().numpy())
    Y_test_pred = data_dict['scaler'].inverse_transform(Y_test_pred)
    
    # Calculate metrics
    train_rmse = root_mean_squared_error(Y_train_actual[:, 0], Y_train_pred[:, 0])
    test_rmse = root_mean_squared_error(Y_test_actual[:, 0], Y_test_pred[:, 0])
    
    # Calculate prediction errors
    prediction_errors = np.abs(Y_test_actual[:, 0] - Y_test_pred[:, 0])
    
    logger.info(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return {
        'Y_train_actual': Y_train_actual[:, 0],
        'Y_test_actual': Y_test_actual[:, 0],
        'Y_test_pred': Y_test_pred[:, 0],
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'prediction_errors': prediction_errors
    }


@app.before_request
def log_request():
    """Log all incoming requests"""
    logger.info(f"Incoming request: {request.method} {request.path}")
    logger.info(f"Headers: {dict(request.headers)}")
    if request.method == 'POST':
        logger.info(f"Body: {request.data}")


@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        html_path = os.path.join(os.path.dirname(__file__), 'index.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return jsonify({'error': 'Could not load index.html'}), 500


@app.route('/style.css')
def serve_css():
    """Serve CSS file"""
    try:
        css_path = os.path.join(os.path.dirname(__file__), 'style.css')
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/css'}
    except Exception as e:
        logger.error(f"Error serving style.css: {str(e)}")
        return '', 404


@app.route('/app.js')
def serve_js():
    """Serve JavaScript file"""
    try:
        js_path = os.path.join(os.path.dirname(__file__), 'app.js')
        with open(js_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'application/javascript'}
    except Exception as e:
        logger.error(f"Error serving app.js: {str(e)}")
        return '', 404


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for stock price predictions
    
    Expected JSON body:
        {
            "ticker": "AAPL"
        }
    """
    try:
        logger.info(f"Received request: {request.method} {request.path}")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Request data: {request.data}")
        
        # Get JSON data
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        request_data = request.get_json()
        if request_data is None:
            logger.error("Could not parse JSON")
            return jsonify({'error': 'Invalid JSON in request body'}), 400
        
        ticker = request_data.get('ticker', '').strip().upper()
        logger.info(f"Requested ticker: {ticker}")
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400
        
        logger.info(f"Processing prediction request for {ticker}")
        
        # Prepare data
        data_dict = prepare_data(ticker)
        
        # Train model
        model = train_model(data_dict['X_train'], data_dict['Y_train'])
        
        # Make predictions
        predictions = make_predictions(model, data_dict)
        
        # Prepare response data
        df = data_dict['df']
        
        # Get dates for test set
        test_start_idx = len(df) - len(predictions['Y_test_actual'])
        test_dates = df.index[test_start_idx:].strftime('%Y-%m-%d').tolist()
        
        # Ensure all values are valid JSON serializable
        def safe_float(x):
            val = float(x)
            if np.isnan(val) or np.isinf(val):
                return 0.0
            return val
        
        # Prepare response
        response_data = {
            'ticker': ticker,
            'train_rmse': safe_float(predictions['train_rmse']),
            'test_rmse': safe_float(predictions['test_rmse']),
            'latest_price': safe_float(df['Close'].iloc[-1]),
            'test_dates': test_dates,
            'test_prices': [safe_float(x) for x in predictions['Y_test_actual']],
            'predicted_prices': [safe_float(x) for x in predictions['Y_test_pred']],
            'prediction_errors': [safe_float(x) for x in predictions['prediction_errors']],
            'all_dates': df.index.strftime('%Y-%m-%d').tolist(),
            'all_prices': [safe_float(x) for x in df['Close'].values]
        }
        
        logger.info(f"Successfully generated predictions for {ticker}")
        logger.info(f"Response data keys: {response_data.keys()}")
        logger.info(f"Response data types: train_rmse={type(response_data['train_rmse'])}, test_rmse={type(response_data['test_rmse'])}")
        return jsonify(response_data), 200
    
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'type': type(e).__name__
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'device': str(device)
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    logger.warning(f"404 Not Found: {request.path}")
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 Method Not Allowed errors"""
    logger.warning(f"405 Method Not Allowed: {request.method} {request.path}")
    return jsonify({'error': f'Method {request.method} not allowed for {request.path}'}), 405


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"500 Internal Server Error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Stock Price Predictor API...")
    logger.info(f"Using device: {device}")
    logger.info("Server running on http://127.0.0.1:5000")
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
