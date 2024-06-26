from flask import Flask, render_template
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("macroeconomics.csv", sep=";")
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/correlation')
def correlation():
    corr_matrix = df.corr()
    corr_fig = px.imshow(corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns, color_continuous_scale='viridis')
    corr_fig.update_layout(title='Correlation Heatmap')
    return corr_fig.to_html()


@app.route('/consumer_price_index_distribution')
def consumer_price_index_distribution():
    fig = px.histogram(df, x='consumer_price_index', nbins=20, title='Distribution of Consumer Price Index')
    return fig.to_html()

@app.route('/unemployment_rate_distribution')
def unemployment_rate_distribution():
    fig = px.histogram(df, x='unemployment_rate', nbins=20, title='Distribution of Unemployment Rate')
    return fig.to_html()

@app.route('/eurpln_vs_usdpln_scatter')
def eurpln_vs_usdpln_scatter():
    fig = px.scatter(df, x='EURPLN', y='USDPLN', title='Scatter plot of EURPLN vs USDPLN')
    return fig.to_html()

@app.route('/reference_rate_nbp_line')
def reference_rate_nbp_line():
    fig = px.line(df, x='date', y='reference_rate_NBP', title='Reference Rate NBP over Time')
    return fig.to_html()

@app.route('/cpi_boxplot_year')
def cpi_boxplot_year():
    df['year'] = df['date'].dt.year
    fig = px.box(df, x='year', y='consumer_price_index', title='Consumer Price Index Boxplot Grouped by Year')
    return fig.to_html()

@app.route('/pairplot_selected_features')
def pairplot_selected_features():
    fig = px.scatter_matrix(df[['consumer_price_index', 'account_balance', 'avg_monthly_salary_enterprise', 'unemployment_rate']], title='Pairplot of Selected Numerical Features')
    return fig.to_html()



if __name__ == '__main__':
    app.run(debug=True)
