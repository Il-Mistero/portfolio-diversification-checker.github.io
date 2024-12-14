import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return "Error: No data received", 400

        tickers = data.get('ticker', [])
        investments = data.get('investment', [])
        period = data.get('period', '1y')

        # Basic validation
        if not tickers or not investments:
            return "Error: Missing tickers or investments", 400

        if len(tickers) != len(investments):
            return "Error: Number of tickers and investments must match", 400

        # Convert investments to float
        try:
            investments = [float(inv) for inv in investments]
        except ValueError:
            return "Error: Invalid investment amounts", 400

        # Download data
        try:
            # Download data for all tickers
            stock_data = yf.download(tickers, period=period)
            
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
                </div>
            '''

            return response_html

        except Exception as e:
            print(f"Error in data processing: {str(e)}")  # Debug log
            return f"Error downloading stock data: {str(e)}", 400

    except Exception as e:
        print(f"Error in analyze route: {str(e)}")  # Debug log
        return f"Server error: {str(e)}", 500

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

