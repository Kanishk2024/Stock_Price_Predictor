let priceChart = null;
let errorChart = null;
let historicalChart = null;

async function predictStock() {
    const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
    
    if (!ticker) {
        showError('Please enter a ticker symbol');
        return;
    }

    // Reset UI
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('predictBtn').disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ticker: ticker })
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        const responseText = await response.text();
        console.log('Raw response:', responseText);
        
        if (!response.ok) {
            try {
                const error = JSON.parse(responseText);
                throw new Error(error.error || `HTTP ${response.status}: Failed to get prediction`);
            } catch (e) {
                if (e instanceof SyntaxError) {
                    throw new Error(`HTTP ${response.status}: ${responseText.substring(0, 200)}`);
                }
                throw e;
            }
        }

        const data = JSON.parse(responseText);
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred. Please try again.');
    } finally {
        document.getElementById('loadingSpinner').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
    }
}

function displayResults(data) {
    // Display metrics
    document.getElementById('trainRmse').textContent = data.train_rmse.toFixed(2);
    document.getElementById('testRmse').textContent = data.test_rmse.toFixed(2);
    document.getElementById('latestPrice').textContent = '$' + data.latest_price.toFixed(2);

    // Show results section
    document.getElementById('resultsSection').style.display = 'block';

    // Draw price chart
    drawPriceChart(data);

    // Draw error chart
    drawErrorChart(data);

    // Draw historical chart
    drawHistoricalChart(data);

    // Scroll to results
    setTimeout(() => {
        document.querySelector('.results-section').scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

function drawPriceChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (priceChart) {
        priceChart.destroy();
    }

    const dates = data.test_dates.slice(0, 100); // Limit to 100 points for clarity
    const actualPrices = data.test_prices.slice(0, 100);
    const predictedPrices = data.predicted_prices.slice(0, 100);

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Actual Price',
                    data: actualPrices,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.3,
                    fill: false,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#10b981',
                    pointBorderColor: '#10b981'
                },
                {
                    label: 'Predicted Price',
                    data: predictedPrices,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.3,
                    fill: false,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#ef4444',
                    pointBorderColor: '#ef4444'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 12 }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        }
    });
}

function drawErrorChart(data) {
    const ctx = document.getElementById('errorChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (errorChart) {
        errorChart.destroy();
    }

    const dates = data.test_dates.slice(0, 100);
    const errors = data.prediction_errors.slice(0, 100);
    const rmseLine = Array(errors.length).fill(data.test_rmse);

    errorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Prediction Error',
                    data: errors,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.3,
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#ef4444'
                },
                {
                    label: 'RMSE Threshold',
                    data: rmseLine,
                    borderColor: '#10b981',
                    borderDash: [5, 5],
                    tension: 0,
                    fill: false,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 12 }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        }
    });
}

function drawHistoricalChart(data) {
    const ctx = document.getElementById('historicalChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (historicalChart) {
        historicalChart.destroy();
    }

    const dates = data.all_dates;
    const prices = data.all_prices;

    historicalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Historical Close Price',
                    data: prices,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.3,
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 12 }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        }
    });
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = '⚠️ ' + message;
    errorDiv.style.display = 'block';
}

// Allow Enter key to submit
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('tickerInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            predictStock();
        }
    });
});
