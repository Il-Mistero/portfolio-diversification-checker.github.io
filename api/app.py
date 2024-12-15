import matplotlib
matplotlib.use('Agg') 

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
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
import os
import logging

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir,
           static_url_path='/static')
CORS(app)

SP500_STOCKS = {}

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_stock_symbols():
    """Fetch stock list from local JSON file"""
    try:
        logger.info("Starting to fetch stock symbols")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'stocks.json')
        
        logger.info(f"Attempting to read from: {json_path}")
        

        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                stocks_data = json.load(file)
                logger.info(f"Successfully loaded JSON data with {len(stocks_data)} entries")
        except FileNotFoundError:
            logger.error(f"stocks.json not found at {json_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {}
        
        # Create dictionary of stocks
        stocks_dict = {}
        for stock in stocks_data:
            if 'symbol' in stock and 'name' in stock:
                symbol = stock['symbol']
                name = stock['name']
                stocks_dict[symbol] = f"{symbol} - {name}"
        
        logger.info(f"Successfully processed {len(stocks_dict)} stocks")
        return stocks_dict

    except Exception as e:
        logger.error(f"Error in get_stock_symbols: {e}", exc_info=True)
        return {}

def get_stock_data(tickers, period):
    try:
        return yf.download(tickers, period=period)
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        return pd.DataFrame()

@app.route('/get_stocks')
def get_stocks():
    try:
        logger.info("Handling /get_stocks request")
        stocks = get_stock_symbols()
        
        if not stocks:
            logger.error("No stocks were fetched")
            return jsonify({"error": "Failed to fetch stocks"}), 500
            
        logger.info(f"Successfully returning {len(stocks)} stocks")
        return jsonify(stocks)
        
    except Exception as e:
        logger.error(f"Error in get_stocks route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    try:
        logger.info("Handling home page request")
        stocks = get_stock_symbols()
        
        if not stocks:
            logger.warning("No stocks available for home page")
            
        return render_template('index.html', stocks=stocks)
        
    except Exception as e:
        logger.error(f"Error in home route: {e}", exc_info=True)
        return "Error loading page", 500

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

        print(f"Processing request for tickers: {tickers}")
        print(f"Investments: {investments}")
        print(f"Period: {period}")

        if not tickers or not investments:
            return "Error: Missing tickers or investments", 400

        if len(tickers) != len(investments):
            return f"Error: Number of tickers ({len(tickers)}) and investments ({len(investments)}) must match", 400

        try:
            investments = [float(inv) for inv in investments]
        except ValueError:
            return "Error: Invalid investment amounts", 400

        stock_data = get_stock_data(tickers, period)
        
        if stock_data.empty:
            return "Error: No data found for the specified tickers", 400

        if isinstance(stock_data, pd.DataFrame):
            if len(tickers) == 1:
                data = stock_data['Adj Close']
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])
            else:
                data = stock_data['Adj Close']

        returns = data.pct_change().dropna()
        
        total_investment = sum(investments)
        weights = [inv / total_investment for inv in investments]

        portfolio_return = np.dot(returns.mean(), weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        plt.style.use('default')
        
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
        
        pie_buffer = BytesIO()
        plt.savefig(pie_buffer, 
                   format='png', 
                   bbox_inches='tight', 
                   dpi=150,
                   pad_inches=0.5,
                   facecolor='#0f3460',
                   edgecolor='none')
        pie_buffer.seek(0)
        pie_chart = base64.b64encode(pie_buffer.getvalue()).decode('utf-8')
        plt.close()

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
        
        plt.xticks(rotation=45, ha='right', color='white', size=12, weight='bold')
        plt.yticks(rotation=0, color='white', size=12, weight='bold')
        
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

        asset_returns = returns.mean() * 252  
        asset_volatility = returns.std() * np.sqrt(252) 

        plt.figure(figsize=(15, 10)) 
        plt.style.use('dark_background')  

        plt.scatter(asset_volatility, asset_returns, 
                   c='#3498db',
                   label='Individual Assets', 
                   s=100)

        plt.scatter(portfolio_volatility, portfolio_return, 
                   c='#e74c3c',  
                   label='Portfolio', 
                   marker='X', 
                   s=200)  

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

        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().tick_params(colors='white')

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

        plt.figure(figsize=(15, 10))
        plt.style.use('dark_background')

        bright_colors = [
            '#FF1E1E',
            '#00FF00',
            '#1E90FF',
            '#FFD700',
            '#FF1493',
            '#00FFFF',
            '#FF8C00',
            '#FF00FF',
            '#7FFF00', 
            '#00FF7F'   
        ]

        for idx, ticker in enumerate(tickers):
            if len(tickers) == 1:
                prices = stock_data['Adj Close']
            else:
                prices = stock_data['Adj Close'][ticker]
            
            normalized_prices = prices * 100 / prices.iloc[0]
            plt.plot(prices.index, normalized_prices, 
                     label=ticker, 
                     linewidth=2.5,  
                     color=bright_colors[idx % len(bright_colors)]) 

        plt.title('Stock Price Trends (Normalized to 100)', pad=20, size=18, color='white', weight='bold')
        plt.xlabel('Date', color='white', size=12, weight='bold')
        plt.ylabel('Normalized Price', color='white', size=12, weight='bold')
        plt.grid(True, alpha=0.2)

        plt.legend(bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  fontsize=12,
                  framealpha=0.8, 
                  edgecolor='white') 

        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().tick_params(colors='white', labelsize=10)

        plt.xticks(rotation=45)

        plt.tight_layout()

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
        import traceback
        print(f"Error in analyze route: {str(e)}")
        print(traceback.format_exc())
        return f"Server error: {str(e)}", 500

@app.route('/search_stocks')
def search_stocks():
    try:
        query = request.args.get('query', '').upper()
        stocks = get_stock_symbols()
        
        filtered_stocks = {
            k: v for k, v in stocks.items() 
            if query in k.upper() or query in v.upper()
        }
        
        return jsonify(filtered_stocks)
    except Exception as e:
        logger.error(f"Error in search_stocks: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)