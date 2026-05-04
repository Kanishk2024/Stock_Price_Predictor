# Stock Price Predictor

A full-stack web application that predicts stock prices using LSTM (Long Short-Term Memory) neural networks. Simply enter a company's ticker symbol and get real-time price predictions with visual analytics.

## Features

✨ **Key Features:**
- **LSTM Neural Network**: Trained on historical stock data from Yahoo Finance
- **Real-time Predictions**: Get predictions for any stock ticker (AAPL, GOOGL, MSFT, etc.)
- **Interactive Charts**: Visualize actual vs predicted prices and prediction errors
- **Performance Metrics**: Training and testing RMSE scores
- **Historical Data**: Complete price history visualization
- **Responsive Design**: Works on desktop and mobile devices
- **GPU Support**: Automatic CUDA support if available

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Optional: NVIDIA GPU with CUDA for faster training

### Step 1: Clone/Download the Project
```bash
cd "Stock Price Predictor"
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Enter a stock ticker** (e.g., AAPL for Apple, GOOGL for Google)
3. **Click "Predict"** or press Enter
4. **Wait for results** - the model will train and generate predictions
5. **View the charts** showing:
   - Actual vs Predicted prices
   - Prediction errors over time
   - Historical price data

## How It Works

### Data Flow
1. **Data Fetching**: Historical stock data is downloaded from Yahoo Finance (2020-01-01 onwards)
2. **Normalization**: Closing prices are normalized using StandardScaler
3. **Sequence Creation**: Data is split into sequences of 30 consecutive trading days
4. **Train/Test Split**: 80% of data for training, 20% for testing
5. **Model Training**: LSTM model is trained with 2 layers, 32 hidden units
6. **Predictions**: The trained model makes predictions on the test set
7. **Visualization**: Results are displayed with interactive charts

### Model Architecture
```
Input (Sequence of 30 prices)
    ↓
LSTM Layer 1 (32 units)
    ↓
LSTM Layer 2 (32 units)
    ↓
Fully Connected Layer (1 unit)
    ↓
Output (Predicted Price)
```

## API Endpoints

### POST /api/predict
Make a stock price prediction

**Request:**
```json
{
    "ticker": "AAPL"
}
```

**Response:**
```json
{
    "ticker": "AAPL",
    "train_rmse": 2.35,
    "test_rmse": 3.42,
    "latest_price": 185.65,
    "test_dates": ["2023-01-01", "2023-01-02", ...],
    "test_prices": [150.5, 151.2, ...],
    "predicted_prices": [150.8, 151.1, ...],
    "prediction_errors": [0.3, 0.1, ...],
    "all_dates": [...],
    "all_prices": [...]
}
```

### GET /api/health
Health check endpoint

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00",
    "device": "cuda" or "cpu"
}
```

## File Structure

```
Stock Price Predictor/
├── app.py                 # Flask backend server
├── Main.py               # Original Jupyter notebook converted to script
├── index.html            # Frontend HTML interface
├── style.css             # Styling for the web interface
├── app.js                # Frontend JavaScript logic
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Performance Metrics

- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
  - Lower RMSE = Better predictions
  - Training RMSE: Error on training data
  - Testing RMSE: Error on unseen test data

## Troubleshooting

### Issue: "No data found for ticker"
- **Solution**: Ensure you're using a valid stock ticker symbol
- Try common ones: AAPL, GOOGL, MSFT, TSLA, AMZN

### Issue: "Port 5000 already in use"
- **Solution**: Either close the application using port 5000, or modify the port in `app.py`:
  ```python
  app.run(debug=True, host='127.0.0.1', port=5001)  # Use port 5001 instead
  ```

### Issue: CUDA out of memory errors
- **Solution**: The code will automatically fall back to CPU if needed

### Issue: Slow predictions
- This is normal on first run as the model trains for 200 epochs
- Subsequent requests for the same or different ticker will use the same process

## Requirements

See `requirements.txt` for all dependencies:
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **yfinance**: Fetching stock data from Yahoo Finance
- **torch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **matplotlib**: Data visualization (optional)
- **flask**: Web framework
- **flask-cors**: Cross-Origin Resource Sharing support

## Disclaimer

⚠️ **Important**: This application is for educational purposes only. Stock predictions are based on historical data and should not be used as sole basis for investment decisions. Always consult with financial advisors before making investment choices.

## Future Enhancements

- [ ] Save trained models for faster predictions
- [ ] Support for multiple time horizons
- [ ] Technical indicators integration
- [ ] Portfolio analysis features
- [ ] User authentication and history tracking
- [ ] Real-time data updates
- [ ] Ensemble models for better accuracy

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please open an issue in the project repository or contact the development team.

---

Happy predicting! 📈
