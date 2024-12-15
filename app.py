import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time  # Add this at the top with other imports

app = Flask(__name__, template_folder="templates")
CORS(app)

# Configure Flask-Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 3600  # Cache for 1 hour
})

def get_sp500_symbols():
    """Fetch S&P 500 symbols and company names"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        df = pd.read_html(str(table))[0]
        # Format the options as "TICKER - Company Name"
        return {row['Symbol']: f"{row['Symbol']} - {row['Security']}" 
                for _, row in df.iterrows()}
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        return {}

# Cache for storing stock data
SP500_STOCKS = {}

@cache.memoize(timeout=3600)
def get_stock_data(tickers, period):
    """Cached function to get stock data"""
    return yf.download(tickers, period=period)

def update_stock_data():
    """Update the S&P 500 stock data"""
    global SP500_STOCKS
    SP500_STOCKS = get_sp500_symbols()
    
    # Save to a JSON file as backup
    with open('static/sp500_stocks.json', 'w') as f:
        json.dump(SP500_STOCKS, f)

# Update Flask routes
@app.route('/get_stocks')
def get_stocks():
    """Get S&P 500 stocks"""
    try:
        # Try to load from memory first
        if not SP500_STOCKS:
            try:
                # Try to load from backup file
                with open('static/sp500_stocks.json', 'r') as f:
                    SP500_STOCKS.update(json.load(f))
            except:
                # If no backup, fetch new data
                update_stock_data()
        
        return jsonify(SP500_STOCKS)
    except Exception as e:
        print(f"Error in get_stocks: {e}")
        return jsonify({})

# Update your existing routes
@app.route('/')
def home():
    # Try to load stock data if not already loaded
    if not SP500_STOCKS:
        try:
            with open('static/sp500_stocks.json', 'r') as f:
                SP500_STOCKS.update(json.load(f))
        except:
            update_stock_data()
    
    # Pass the stocks data to the template
    return render_template('index.html', stocks=SP500_STOCKS)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        print("Received data:", data)
        
        if not data:
            return "Error: No data received", 400

        tickers = data.get('ticker', [])
        investments = data.get('investment', [])
        period = data.get('period', '1y')

        if not tickers or not investments:
            return "Error: Missing tickers or investments", 400

        if len(tickers) != len(investments):
            return f"Error: Number of tickers ({len(tickers)}) and investments ({len(investments)}) must match", 400

        try:
            investments = [float(inv) for inv in investments]
        except ValueError:
            return "Error: Invalid investment amounts", 400

        # Use cached stock data
        stock_data = get_stock_data(tickers, period)
        
        if stock_data.empty:
            return "Error: No data found for the specified tickers", 400

        # Handle 'Adj Close' data
        if isinstance(stock_data, pd.DataFrame):
            if len(tickers) == 1:
                # For single ticker, extract Adj Close and name the column
                data = stock_data['Adj Close']
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])
            else:
                # For multiple tickers, get the Adj Close columns
                data = stock_data['Adj Close']

        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate weights
        total_investment = sum(investments)
        weights = [inv / total_investment for inv in investments]

        # Portfolio calculations
        portfolio_return = np.dot(returns.mean(), weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        # Generate plots
        plt.style.use('default')
        
        # Create pie chart with wider dimensions
        plt.figure(figsize=(15, 10))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c']
        plt.pie(weights, 
               labels=tickers, 
               autopct='%1.1f%%', 
               startangle=90, 
               textprops={'fontsize': 14, 'color': 'white', 'weight': 'bold'},
               radius=1.2,
               labeldistance=1.05,
               colors=colors,
               pctdistance=0.85)
        plt.title('Portfolio Allocation', pad=20, size=18, y=1.05, color='white', weight='bold')
        
        # Save with dark background
        pie_buffer = BytesIO()
        plt.savefig(pie_buffer, 
                   format='png', 
                   bbox_inches='tight', 
                   dpi=150,
                   pad_inches=0.5,
                   facecolor='#0f3460',  # Dark background
                   edgecolor='none')
        pie_buffer.seek(0)
        pie_chart = base64.b64encode(pie_buffer.getvalue()).decode('utf-8')
        plt.close()

        # Create correlation heatmap with wider dimensions
        plt.figure(figsize=(20, 10))
        sns.set_theme(style="dark")
        heatmap = sns.heatmap(returns.corr(), 
                            annot=True,
                            cmap='coolwarm',
                            fmt='.2f',
                            square=True,
                            cbar_kws={'shrink': .3, 'label': 'Correlation Coefficient'},
                            annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'},
                            center=0,
                            vmin=-1, vmax=1)
        
        plt.title('Correlation Matrix', pad=20, size=18, color='white', weight='bold')
        
        # Make labels more visible
        plt.xticks(rotation=45, ha='right', color='white', size=12, weight='bold')
        plt.yticks(rotation=0, color='white', size=12, weight='bold')
        
        # Adjust colorbar ticks
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(colors='white', labelsize=12)
        
        plt.tight_layout()
        
        corr_buffer = BytesIO()
        plt.savefig(corr_buffer, 
                   format='png', 
                   bbox_inches='tight', 
                   dpi=150,
                   pad_inches=0.5,
                   facecolor='#0f3460',
                   edgecolor='none')
        corr_buffer.seek(0)
        corr_chart = base64.b64encode(corr_buffer.getvalue()).decode('utf-8')
        plt.close()

        # Calculate individual asset metrics
        asset_returns = returns.mean() * 252  # Annualized returns
        asset_volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Create risk-return scatter plot
        plt.figure(figsize=(15, 10))  # Matching the size of your other plots
        plt.style.use('dark_background')  # To match your dark theme

        plt.scatter(asset_volatility, asset_returns, 
                   c='#3498db',  # Blue color matching your theme
                   label='Individual Assets', 
                   s=100)  # Increased marker size

        plt.scatter(portfolio_volatility, portfolio_return, 
                   c='#e74c3c',  # Red color matching your theme
                   label='Portfolio', 
                   marker='X', 
                   s=200)  # Increased marker size

        # Add labels for each asset
        for i, txt in enumerate(tickers):
            plt.annotate(txt, 
                        (asset_volatility[i], asset_returns[i]),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        color='white',
                        fontsize=10,
                        fontweight='bold')

        plt.xlabel('Volatility (Standard Deviation)', color='white', size=12, weight='bold')
        plt.ylabel('Expected Return', color='white', size=12, weight='bold')
        plt.title('Risk vs Return Analysis', pad=20, size=18, color='white', weight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.2)

        # Style the axes
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().tick_params(colors='white')

        # Save the plot
        risk_return_buffer = BytesIO()
        plt.savefig(risk_return_buffer, 
                    format='png', 
                    bbox_inches='tight', 
                    dpi=150,
                    pad_inches=0.5,
                    facecolor='#0f3460',
                    edgecolor='none')
        risk_return_buffer.seek(0)
        risk_return_chart = base64.b64encode(risk_return_buffer.getvalue()).decode('utf-8')
        plt.close()

        # Create stock price trend plot
        plt.figure(figsize=(15, 10))
        plt.style.use('dark_background')

        # Define bright, distinct colors
        bright_colors = [
            '#FF1E1E',  # Bright Red
            '#00FF00',  # Bright Green
            '#1E90FF',  # Bright Blue
            '#FFD700',  # Gold
            '#FF1493',  # Deep Pink
            '#00FFFF',  # Cyan
            '#FF8C00',  # Dark Orange
            '#FF00FF',  # Magenta
            '#7FFF00',  # Chartreuse
            '#00FF7F'   # Spring Green
        ]

        # Plot each stock's price with bright colors
        for idx, ticker in enumerate(tickers):
            if len(tickers) == 1:
                prices = stock_data['Adj Close']
            else:
                prices = stock_data['Adj Close'][ticker]
            
            # Normalize prices to start at 100 for better comparison
            normalized_prices = prices * 100 / prices.iloc[0]
            plt.plot(prices.index, normalized_prices, 
                     label=ticker, 
                     linewidth=2.5,  # Made lines slightly thicker
                     color=bright_colors[idx % len(bright_colors)])  # Cycle through bright colors

        plt.title('Stock Price Trends (Normalized to 100)', pad=20, size=18, color='white', weight='bold')
        plt.xlabel('Date', color='white', size=12, weight='bold')
        plt.ylabel('Normalized Price', color='white', size=12, weight='bold')
        plt.grid(True, alpha=0.2)

        # Enhance legend visibility
        plt.legend(bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  fontsize=12,
                  framealpha=0.8,  # Semi-transparent background
                  edgecolor='white')  # White edge around legend

        # Style the axes
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().tick_params(colors='white', labelsize=10)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot
        price_trend_buffer = BytesIO()
        plt.savefig(price_trend_buffer, 
                    format='png', 
                    bbox_inches='tight', 
                    dpi=150,
                    pad_inches=0.5,
                    facecolor='#0f3460',
                    edgecolor='none')
        price_trend_buffer.seek(0)
        price_trend_chart = base64.b64encode(price_trend_buffer.getvalue()).decode('utf-8')
        plt.close()

        response_html = f'''
            <div class="container-fluid mt-4">
                <h3 class="mb-4 text-center">Portfolio Analysis Results</h3>
                
                <!-- Key Metrics Card -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title text-center">Key Metrics</h5>
                                <div class="row text-center">
                                    <div class="col-md-4">
                                        <p class="card-text">Expected Annual Return: {portfolio_return:.2%}</p>
                                    </div>
                                    <div class="col-md-4">
                                        <p class="card-text">Portfolio Volatility: {portfolio_volatility:.2%}</p>
                                    </div>
                                    <div class="col-md-4">
                                        <p class="card-text">Sharpe Ratio: {sharpe_ratio:.2f}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Risk-Return Analysis Card -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Risk-Return Analysis</h5>
                                <div class="d-flex justify-content-center">
                                    <img src="data:image/png;base64,{risk_return_chart}" 
                                         class="img-fluid" 
                                         alt="Risk-Return Analysis"
                                         style="max-width:90%; width: 700px; max-height: 90%; height: 500px; object-fit: contain;">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Portfolio Allocation Card -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Portfolio Allocation</h5>
                                <div class="d-flex justify-content-center">
                                    <img src="data:image/png;base64,{pie_chart}" 
                                         class="img-fluid" 
                                         alt="Portfolio Allocation"
                                         style="max-width:90%; width: 700px; max-height: 90%;height: 500px; object-fit: contain;">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Correlation Matrix Card -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Correlation Matrix</h5>
                                <div class="d-flex justify-content-center">
                                    <img src="data:image/png;base64,{corr_chart}" 
                                         class="img-fluid" 
                                         alt="Correlation Matrix"
                                         style="max-width:90%; width: 700px; max-height: 90%; height: 700px; object-fit: contain;">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Price Trends Card -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Price Trends (1 Year)</h5>
                                <div class="d-flex justify-content-center">
                                    <img src="data:image/png;base64,{price_trend_chart}" 
                                         class="img-fluid" 
                                         alt="Price Trends"
                                         style="max-width:90%; width: 700px; max-height: 90%; height: 500px; object-fit: contain;">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        '''

        return response_html

    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        return f"Server error: {str(e)}", 500

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
