import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from prophet import Prophet
from datetime import datetime, timedelta
import os
import io

# Set page configuration
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-text {
        background-color: #e8f4f8;
        border-left: 5px solid #2196F3;
        padding: 10px 15px;
        margin-bottom: 20px;
        border-radius: 0 5px 5px 0;
    }
    .stPlotlyChart {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('/Users/arjavkasliwal/Downloads/finaldatasetecom.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        return df
    except FileNotFoundError:
        st.error("Cannot find finaldatasetecom.csv. Please upload a file.")
        return None

# Function to load preprocessed e-commerce data
@st.cache_data
def load_ecommerce_data():
    try:
        df = pd.read_csv('/Users/arjavkasliwal/Downloads/finaldatasetecom.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        return df
    except FileNotFoundError:
        return None

# Function to load sales prediction model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_sales_prediction_model.pkl')
        with open('model_features.txt', 'r') as f:
            features = f.read().splitlines()
        return model, features
    except:
        return None, None

# Function to load category forecasts
@st.cache_data
def load_category_forecasts():
    try:
        forecasts = pd.read_csv('category_forecasts.csv')
        return forecasts
    except FileNotFoundError:
        return None

# Sidebar for navigation
st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select Page",
    ["Executive Overview", "Sales Analytics", "Product Trends", "Brand Analytics", 
     "Customer Insights", "Inventory Management", "Sales Prediction", "Sustainability"]
)

# Load all data
df = load_data()
ecommerce_data = load_ecommerce_data()
model, model_features = load_model()
category_forecasts = load_category_forecasts()

# Allow file upload if data is missing
if df is None:
    uploaded_file = st.sidebar.file_uploader("Upload your supply chain data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Order Date'] = pd.to_datetime(df['Order Date'])

# Main dashboard starts here
if df is not None:
    # Prepare common data
    df['YearMonth'] = df['Order Date'].dt.to_period('M')
    monthly_sales = df.groupby('YearMonth')['Sales'].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()
    
    total_sales = df['Sales'].sum()
    total_orders = df['Order Date'].nunique()
    avg_order_value = total_sales / total_orders
    
    # Calculate additional metrics
    if 'Profit' not in df.columns and 'Cost_Price' in df.columns and 'Quantity' in df.columns:
        df['Profit'] = df['Sales'] - (df['Cost_Price'] * df['Quantity'])
    
    if 'Profit' in df.columns:
        total_profit = df['Profit'].sum()
        profit_margin = (total_profit / total_sales) * 100
    else:
        total_profit = "N/A"
        profit_margin = "N/A"
    
    # EXECUTIVE OVERVIEW PAGE
    if page == "Executive Overview":
        st.markdown("<div class='main-header'>Executive Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='insight-text' style='color: black;'>This dashboard provides a high-level overview of your e-commerce business performance, highlighting key metrics and trends.</div>", unsafe_allow_html=True)
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", f"₹{total_sales:,.2f}")
        
        with col2:
            st.metric("Total Orders", f"{total_orders:,}")
        
        with col3:
            st.metric("Average Order Value", f"₹{avg_order_value:,.2f}")
        
        with col4:
            if isinstance(profit_margin, str):
                st.metric("Profit Margin", profit_margin)
            else:
                st.metric("Profit Margin", f"{profit_margin:.2f}%")
        
        # Sales trend chart
        st.markdown("<div class='sub-header'>Sales Trend</div>", unsafe_allow_html=True)
        
        fig = px.line(
            x=monthly_sales.index, 
            y=monthly_sales.values,
            labels={"x": "Date", "y": "Sales (₹)"},
            title="Monthly Sales Trend"
        )
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Sales (₹)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top products and categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='sub-header'>Top 10 Products</div>", unsafe_allow_html=True)
            product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=product_sales.values, 
                y=product_sales.index,
                orientation='h',
                labels={"x": "Sales (₹)", "y": "Product"},
                title="Top 10 Products by Sales"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<div class='sub-header'>Category Performance</div>", unsafe_allow_html=True)
            category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
            
            fig = px.pie(
                names=category_sales.index, 
                values=category_sales.values,
                title="Sales by Category",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Future sales forecast
        st.markdown("<div class='sub-header'>Sales Forecast</div>", unsafe_allow_html=True)
        
        if category_forecasts is not None:
            top_categories = category_forecasts.columns[:-1]  # Exclude date column
            selected_categories = st.multiselect(
                "Select categories to display forecast",
                options=top_categories,
                default=top_categories[:3]
            )
            
            if selected_categories:
                forecast_df = category_forecasts.copy()
                forecast_df = forecast_df[['Date'] + [c for c in selected_categories if c in forecast_df.columns]]
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                
                fig = go.Figure()
                for category in selected_categories:
                    if f"{category}_forecast" in forecast_df.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'], 
                            y=forecast_df[f"{category}_forecast"],
                            mode='lines',
                            name=category
                        ))
                
                fig.update_layout(
                    title="12-Month Sales Forecast by Category",
                    xaxis_title="Date",
                    yaxis_title="Forecasted Sales (₹)",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # SALES ANALYTICS PAGE
    
    elif page == "Sales Analytics":
        st.markdown("<div class='main-header'>Sales Analytics</div>", unsafe_allow_html=True)
        
        # Time period selector
        time_options = ["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months"]
        time_period = st.selectbox("Select Time Period", time_options)
        
        if time_period == "Last 12 Months":
            filtered_df = df[df['Order Date'] >= df['Order Date'].max() - pd.DateOffset(months=12)]
        elif time_period == "Last 6 Months":
            filtered_df = df[df['Order Date'] >= df['Order Date'].max() - pd.DateOffset(months=6)]
        elif time_period == "Last 3 Months":
            filtered_df = df[df['Order Date'] >= df['Order Date'].max() - pd.DateOffset(months=3)]
        else:
            filtered_df = df
        
        # Sales metrics
        col1, col2, col3, col4 = st.columns(4)
        
        period_sales = filtered_df['Sales'].sum()
        period_orders = filtered_df['Order Date'].nunique()
        period_avg_value = period_sales / period_orders if period_orders > 0 else 0
        
        with col1:
            st.metric("Period Sales", f"₹{period_sales:,.2f}")
        
        with col2:
            st.metric("Period Orders", f"{period_orders:,}")
        
        with col3:
            st.metric("Avg Order Value", f"₹{period_avg_value:,.2f}")
        
        with col4:
            if 'Quantity' in filtered_df.columns:
                avg_units = filtered_df['Quantity'].mean()
                st.metric("Avg Units per Order", f"{avg_units:.2f}")
            else:
                st.metric("Avg Units per Order", "N/A")
        
        # Sales breakdown tabs
        tab1, tab2, tab3 = st.tabs(["Time Analysis", "Geographic Analysis", "Channel Analysis"])
        
        with tab1:
            # Time series decomposition
            st.markdown("<div class='sub-header'>Sales Time Analysis</div>", unsafe_allow_html=True)
            
            # Choose time granularity
            time_grain = st.radio("Select Time Granularity", ["Monthly", "Weekly", "Daily"], horizontal=True)
            
            if time_grain == "Monthly":
                filtered_df['TimePeriod'] = filtered_df['Order Date'].dt.to_period('M')
            elif time_grain == "Weekly":
                filtered_df['TimePeriod'] = filtered_df['Order Date'].dt.to_period('W')
            else:
                filtered_df['TimePeriod'] = filtered_df['Order Date'].dt.to_period('D')
            
            time_sales = filtered_df.groupby('TimePeriod')['Sales'].sum()
            time_sales.index = time_sales.index.to_timestamp()
            
            # Plot time series
            fig = px.line(
                x=time_sales.index, 
                y=time_sales.values,
                labels={"x": "Date", "y": "Sales ($)"},
                title=f"{time_grain} Sales Trend"
            )
            
            if time_grain == "Monthly":
                # Add trendline
                fig.add_trace(
                    go.Scatter(
                        x=time_sales.index, 
                        y=time_sales.rolling(window=3).mean(),
                        mode='lines',
                        name='3-Month Moving Average',
                        line=dict(color='red', dash='dash')
                    )
                )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Day of week analysis if we have enough data
            if time_grain == "Daily" and len(filtered_df) > 30:
                # Add day of week analysis
                filtered_df['DayOfWeek'] = filtered_df['Order Date'].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_sales = filtered_df.groupby('DayOfWeek')['Sales'].sum()
                dow_sales = dow_sales.reindex(day_order)
                
                fig = px.bar(
                    x=dow_sales.index, 
                    y=dow_sales.values,
                    labels={"x": "Day of Week", "y": "Sales ($)"},
                    title="Sales by Day of Week"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Geographic analysis
            st.markdown("<div class='sub-header'>Geographic Sales Distribution</div>", unsafe_allow_html=True)
            
            if 'State' in filtered_df.columns:
                state_sales = filtered_df.groupby('State')['Sales'].sum().sort_values(ascending=False)
                
                fig = px.choropleth(
                    locations=state_sales.index,
                    locationmode="USA-states",
                    color=state_sales,
                    scope="usa",
                    labels={"color": "Sales ($)", "locations": "State"},
                    title="Sales by State"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top states table
                top_states = state_sales.head(10).reset_index()
                top_states.columns = ['State', 'Sales (₹)']
                top_states['Sales (₹)'] = top_states['Sales (₹)'].map('${:,.2f}'.format)
                
                st.markdown("<div class='sub-header'>Top 10 States by Sales</div>", unsafe_allow_html=True)
                st.table(top_states)
            else:
                st.info("Geographic data not available in the dataset")
        
        with tab3:
            # Sales Channel Analysis
            st.markdown("<div class='sub-header'>Sales Channel Analysis</div>", unsafe_allow_html=True)
            
            if 'Sales_Channel' in filtered_df.columns:
                channel_sales = filtered_df.groupby('Sales_Channel')['Sales'].sum().sort_values(ascending=False)
                
                fig = px.pie(
                    names=channel_sales.index, 
                    values=channel_sales.values,
                    title="Sales by Channel",
                    hole=0.4
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Channel performance over time
                channel_time = filtered_df.groupby(['TimePeriod', 'Sales_Channel'])['Sales'].sum().unstack()
                channel_time.index = channel_time.index.to_timestamp()
                
                fig = px.line(
                    channel_time,
                    x=channel_time.index,
                    y=channel_time.columns,
                    labels={"x": "Date", "y": "Sales (₹)", "variable": "Channel"},
                    title="Channel Performance Over Time"
                )
                fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sales channel data not available in the dataset")
                
    # PRODUCT TRENDS PAGE
    elif page == "Product Trends":
        st.markdown("<div class='main-header'>Product Trends</div>", unsafe_allow_html=True)
        
        # Product trend analysis options
        trend_type = st.radio(
            "Select Trend Analysis",
            ["Top Products", "Category Trends", "Seasonal Patterns", "Product Growth"],
            horizontal=True
        )
        
        if trend_type == "Top Products":
            st.markdown("<div class='sub-header'>Top Performing Products</div>", unsafe_allow_html=True)
            
            # Select performance metric
            metric = st.selectbox(
                "Select Performance Metric", 
                ["Sales", "Quantity", "Profit"] if "Profit" in df.columns else ["Sales", "Quantity"]
            )
            
            # Get top products by selected metric
            if metric in df.columns:
                product_performance = df.groupby('Product')[metric].sum().sort_values(ascending=False)
                top_n = st.slider("Number of Products to Show", 5, 50, 20)
                top_products = product_performance.head(top_n)
                
                fig = px.bar(
                    x=top_products.values, 
                    y=top_products.index,
                    orientation='h',
                    labels={"x": f"{metric} (₹)" if metric != "Quantity" else metric, "y": "Product"},
                    title=f"Top {top_n} Products by {metric}"
                )
                fig.update_layout(height=800, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Products table
                top_products_df = top_products.reset_index()
                if metric == "Sales" or metric == "Profit":
                    top_products_df[metric] = top_products_df[metric].map('₹{:,.2f}'.format)
                else:
                    top_products_df[metric] = top_products_df[metric].map('{:,}'.format)
                
                st.dataframe(top_products_df, use_container_width=True)
        
        elif trend_type == "Category Trends":
            st.markdown("<div class='sub-header'>Category Performance Trends</div>", unsafe_allow_html=True)
            
            # Monthly sales by category
            df['YearMonth'] = df['Order Date'].dt.to_period('M')
            category_monthly = df.groupby(['YearMonth', 'Category'])['Sales'].sum().unstack()
            category_monthly.index = category_monthly.index.to_timestamp()
            
            # Select categories to display
            all_categories = df['Category'].unique()
            selected_categories = st.multiselect(
                "Select Categories to Display", 
                options=all_categories,
                default=all_categories[:5] if len(all_categories) > 5 else all_categories
            )
            
            if selected_categories:
                # Filter category data
                filtered_categories = [c for c in selected_categories if c in category_monthly.columns]
                cat_data = category_monthly[filtered_categories]
                
                # Plot category trends
                fig = px.line(
                    cat_data,
                    x=cat_data.index,
                    y=cat_data.columns,
                    labels={"x": "Date", "y": "Sales (₹)", "variable": "Category"},
                    title="Monthly Sales by Category"
                )
                fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                # Category growth rates
                st.markdown("<div class='sub-header'>Category Growth Rates</div>", unsafe_allow_html=True)
                
                # Calculate growth rates
                category_growth = pd.DataFrame()
                for category in filtered_categories:
                    # Calculate month-over-month growth
                    category_growth[category] = cat_data[category].pct_change() * 100
                
                # Calculate 3-month average growth rate
                category_growth_3m = category_growth.rolling(3).mean()
                
                # Plot category growth rates (3-month average)
                fig = px.line(
                    category_growth_3m.iloc[-12:],
                    x=category_growth_3m.iloc[-12:].index,
                    y=category_growth_3m.iloc[-12:].columns,
                    labels={"x": "Date", "y": "Growth Rate (%)", "variable": "Category"},
                    title="3-Month Rolling Average Growth Rate by Category"
                )
                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="red")
                fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                # Display category forecasts if available
                if category_forecasts is not None:
                    st.markdown("<div class='sub-header'>Category Sales Forecast</div>", unsafe_allow_html=True)
                    
                    forecast_cols = [f"{c}_forecast" for c in filtered_categories if f"{c}_forecast" in category_forecasts.columns]
                    if forecast_cols:
                        forecast_df = category_forecasts[['Date'] + forecast_cols].copy()
                        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                        
                        fig = go.Figure()
                        for col in forecast_cols:
                            category_name = col.replace("_forecast", "")
                            fig.add_trace(go.Scatter(
                                x=forecast_df['Date'], 
                                y=forecast_df[col],
                                mode='lines',
                                name=category_name
                            ))
                        
                        fig.update_layout(
                            title="12-Month Sales Forecast by Category",
                            xaxis_title="Date",
                            yaxis_title="Forecasted Sales (₹)",
                            height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        elif trend_type == "Seasonal Patterns":
            st.markdown("<div class='sub-header'>Seasonal Product Patterns</div>", unsafe_allow_html=True)
            
            # Create seasonal analysis
            seasonal_df = df.copy()
            seasonal_df['Month'] = seasonal_df['Order Date'].dt.month
            seasonal_df['Quarter'] = seasonal_df['Order Date'].dt.quarter
            
            # Select seasonal grouping
            seasonal_group = st.selectbox("Group By", ["Category", "Brand"])
            
            if seasonal_group in df.columns:
                # Create a pivot table
                category_by_month = seasonal_df.groupby([seasonal_group, 'Month'])['Sales'].sum().unstack()
                
                # Format month names
                month_names = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
                category_by_month.columns = [month_names[m] for m in category_by_month.columns]
                
                # Normalize to show seasonal patterns
                normalized_category = category_by_month.div(category_by_month.sum(axis=1), axis=0)
                
                # Select how many items to show
                top_n = st.slider("Number of Items to Show", 5, 30, 15)
                
                # Get top categories/brands by total sales
                top_items = df.groupby(seasonal_group)['Sales'].sum().sort_values(ascending=False).head(top_n).index
                filtered_data = normalized_category.loc[top_items]
                
                # Plot heatmap
                fig = px.imshow(
                    filtered_data,
                    labels=dict(x="Month", y=seasonal_group, color="Normalized Sales"),
                    x=filtered_data.columns,
                    y=filtered_data.index,
                    color_continuous_scale="YlGnBu",
                    title=f"Seasonal {seasonal_group} Sales Patterns (Normalized)"
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
                
                # Identify seasonal winners
                st.markdown("<div class='sub-header'>Seasonal Winners</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Find winter winners (Q4-Q1)
                    winter_data = seasonal_df[seasonal_df['Month'].isin([12, 1, 2])]
                    winter_winners = winter_data.groupby(seasonal_group)['Sales'].sum().sort_values(ascending=False).head(5)
                    
                    st.markdown("#### Winter Winners (Dec-Feb)")
                    winter_df = winter_winners.reset_index()
                    winter_df['Sales'] = winter_df['Sales'].map('₹{:,.2f}'.format)
                    st.table(winter_df)
                    
                    # Find summer winners (Q2-Q3)
                    summer_data = seasonal_df[seasonal_df['Month'].isin([6, 7, 8])]
                    summer_winners = summer_data.groupby(seasonal_group)['Sales'].sum().sort_values(ascending=False).head(5)
                    
                    st.markdown("#### Summer Winners (Jun-Aug)")
                    summer_df = summer_winners.reset_index()
                    summer_df['Sales'] = summer_df['Sales'].map('₹{:,.2f}'.format)
                    st.table(summer_df)
                    
                with col2:
                    # Find spring winners
                    spring_data = seasonal_df[seasonal_df['Month'].isin([3, 4, 5])]
                    spring_winners = spring_data.groupby(seasonal_group)['Sales'].sum().sort_values(ascending=False).head(5)
                    
                    st.markdown("#### Spring Winners (Mar-May)")
                    spring_df = spring_winners.reset_index()
                    spring_df['Sales'] = spring_df['Sales'].map('₹{:,.2f}'.format)
                    st.table(spring_df)
                    
                    # Find fall winners
                    fall_data = seasonal_df[seasonal_df['Month'].isin([9, 10, 11])]
                    fall_winners = fall_data.groupby(seasonal_group)['Sales'].sum().sort_values(ascending=False).head(5)
                    
                    st.markdown("#### Fall Winners (Sep-Nov)")
                    fall_df = fall_winners.reset_index()
                    fall_df['Sales'] = fall_df['Sales'].map('₹{:,.2f}'.format)
                    st.table(fall_df)
        
        elif trend_type == "Product Growth":
            st.markdown("<div class='sub-header'>Product Growth Analysis</div>", unsafe_allow_html=True)
            
            # Calculate growth rates for individual products
            df['YearMonth'] = df['Order Date'].dt.to_period('M')
            
            # Get top products by sales volume
            top_n = st.slider("Number of Top Products to Analyze", 10, 100, 50)
            top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(top_n).index
            
            # Filter to top products and prepare monthly data
            top_products_df = df[df['Product'].isin(top_products)]
            product_monthly = top_products_df.groupby(['YearMonth', 'Product'])['Sales'].sum().unstack()
            
            # Calculate period to analyze
            period_options = {
                "Last 3 Months": 3,
                "Last 6 Months": 6,
                "Last 12 Months": 12
            }
            period = st.selectbox("Growth Period to Analyze", list(period_options.keys()))
            months = period_options[period]
            
            # Calculate growth for each product
            product_growth = {}
            for product in top_products:
                if product in product_monthly.columns:
                    # Get first and last months that have data
                    product_data = product_monthly[product].dropna()
                    if len(product_data) >= 2:
                        # Calculate growth from start to end
                        start_value = product_data.iloc[-months] if len(product_data) >= months else product_data.iloc[0]
                        if start_value > 0:  # Avoid division by zero
                            end_value = product_data.iloc[-1]
                            growth = ((end_value - start_value) / start_value) * 100
                            product_growth[product] = growth
            
            # Create dataframe from growth dictionary
            growth_df = pd.DataFrame(list(product_growth.items()), columns=['Product', 'Growth'])
            growth_df = growth_df.sort_values('Growth', ascending=False)
            
            # Display fastest growing products
            st.markdown(f"#### Fastest Growing Products ({period})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Fastest growing
                fastest_growing = growth_df.head(10)
                fig = px.bar(
                    fastest_growing,
                    x='Growth', 
                    y='Product',
                    labels={"Growth": "Growth (%)", "Product": "Product"},
                    title=f"Top 10 Fastest Growing Products ({period})"
                )
                fig.update_traces(marker_color='green')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Declining products
                declining = growth_df.tail(10).sort_values('Growth')
                fig = px.bar(
                    declining,
                    x='Growth', 
                    y='Product',
                    labels={"Growth": "Growth (%)", "Product": "Product"},
                    title=f"Top 10 Declining Products ({period})"
                )
                fig.update_traces(marker_color='red')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display growth rate distribution
            st.markdown("#### Growth Rate Distribution")
            
            fig = px.histogram(
                growth_df,
                x='Growth',
                nbins=20,
                labels={"Growth": "Growth Rate (%)"},
                title=f"Distribution of Product Growth Rates ({period})"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed product growth table
            st.markdown("#### Detailed Product Growth Analysis")
            
            # Get sales values for first and last period
            product_sales = {}
            for product in growth_df['Product']:
                if product in product_monthly.columns:
                    product_data = product_monthly[product].dropna()
                    if len(product_data) >= months:
                        start_value = product_data.iloc[-months]
                        end_value = product_data.iloc[-1]
                        product_sales[product] = {
                            'Start Value': start_value,
                            'End Value': end_value
                        }
            
            # Create detailed table
            detailed_growth = growth_df.copy()
            detailed_growth['Start Value'] = detailed_growth['Product'].map(lambda p: product_sales.get(p, {}).get('Start Value', 0))
            detailed_growth['End Value'] = detailed_growth['Product'].map(lambda p: product_sales.get(p, {}).get('End Value', 0))
            
            # Format columns
            detailed_growth['Start Value'] = detailed_growth['Start Value'].map('₹{:,.2f}'.format)
            detailed_growth['End Value'] = detailed_growth['End Value'].map('₹{:,.2f}'.format)
            detailed_growth['Growth'] = detailed_growth['Growth'].map('{:,.2f}%'.format)
            
            st.dataframe(detailed_growth, use_container_width=True)
    
    # BRAND ANALYTICS PAGE
    elif page == "Brand Analytics":
        st.markdown("<div class='main-header'>Brand Analytics</div>", unsafe_allow_html=True)
        
        if 'Brand' in df.columns:
            # Brand overview metrics
            brand_sales = df.groupby('Brand')['Sales'].sum().sort_values(ascending=False)
            total_brands = len(brand_sales)
            avg_brand_sales = brand_sales.mean()
            top_brand = brand_sales.index[0]
            top_brand_sales = brand_sales.iloc[0]
            top_brand_share = (top_brand_sales / brand_sales.sum()) * 100
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Brands", f"{total_brands}")
            
            with col2:
                st.metric("Avg Brand Sales", f"₹{avg_brand_sales:,.2f}")
            
            with col3:
                st.metric("Top Brand", f"{top_brand}")
            
            with col4:
                st.metric("Top Brand Market Share", f"{top_brand_share:.2f}%")
            
            # Brand analysis tabs
            tab1, tab2, tab3 = st.tabs(["Brand Performance", "Brand Growth", "Brand Comparison"])
            
            with tab1:
                st.markdown("<div class='sub-header'>Brand Performance Overview</div>", unsafe_allow_html=True)
                
                # Top brands by sales
                top_n = st.slider("Number of Top Brands to Show", 5, 30, 15)
                top_brands = brand_sales.head(top_n)
                
                fig = px.bar(
                    x=top_brands.values, 
                    y=top_brands.index,
                    orientation='h',
                    labels={"x": "Sales (₹)", "y": "Brand"},
                    title=f"Top {top_n} Brands by Sales"
                )
                fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Brand market share
                fig = px.pie(
                    names=brand_sales.head(10).index, 
                    values=brand_sales.head(10).values,
                    title="Top 10 Brands Market Share",
                    hole=0.4
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # If profit data is available, add profitability analysis
                if 'Profit' in df.columns:
                    st.markdown("<div class='sub-header'>Brand Profitability</div>", unsafe_allow_html=True)
                    
                    brand_profit = df.groupby('Brand')['Profit'].sum().sort_values(ascending=False)
                    brand_margin = df.groupby('Brand').apply(lambda x: (x['Profit'].sum() / x['Sales'].sum()) * 100)
                    
                    # Create brand profitability metrics
                    brand_metrics = pd.DataFrame({
                        'Sales': brand_sales,
                        'Profit': brand_profit,
                        'Margin': brand_margin
                    }).sort_values('Sales', ascending=False).head(top_n)
                    
                    # Plot brand margin comparison
                    fig = px.bar(
                        x=brand_metrics.index, 
                        y=brand_metrics['Margin'],
                        labels={"x": "Brand", "y": "Profit Margin (%)"},
                        title=f"Profit Margin by Brand (Top {top_n})"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot margin vs sales bubble chart
                    fig = px.scatter(
                        x=brand_metrics['Sales'], 
                        y=brand_metrics['Margin'],
                        size=brand_metrics['Profit'].abs(),  # Ensure size values are non-negative
                        hover_name=brand_metrics.index,
                        labels={"x": "Sales (₹)", "y": "Profit Margin (%)", "size": "Profit (₹)"},
                        title="Brand Performance Matrix: Sales vs. Margin"
                    )

                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("<div class='sub-header'>Brand Growth Analysis</div>", unsafe_allow_html=True)
                
                # Prepare monthly brand sales
                df['YearMonth'] = df['Order Date'].dt.to_period('M')
                brand_monthly = df.groupby(['YearMonth', 'Brand'])['Sales'].sum().unstack()
                brand_monthly.index = brand_monthly.index.to_timestamp()
                
                # Select brands to display
                top_brands_list = brand_sales.head(20).index.tolist()
                selected_brands = st.multiselect(
                    "Select Brands to Display", 
                    options=top_brands_list,
                    default=top_brands_list[:5]
                )
                
                if selected_brands:
                    # Filter selected brands
                    filtered_brands = [b for b in selected_brands if b in brand_monthly.columns]
                    brand_data = brand_monthly[filtered_brands]
                    
                    # Plot brand trends
                    fig = px.line(
                        brand_data,
                        x=brand_data.index,
                        y=brand_data.columns,
                        labels={"x": "Date", "y": "Sales (₹)", "variable": "Brand"},
                        title="Monthly Sales by Brand"
                    )
                    fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate brand growth over selected period
                    period_options = {
                        "Last 3 Months": 3,
                        "Last 6 Months": 6,
                        "Last 12 Months": 12
                    }
                    period = st.selectbox("Brand Growth Period to Analyze", list(period_options.keys()))
                    months = period_options[period]
                    
                    # Calculate growth for each brand
                    brand_growth = {}
                    for brand in filtered_brands:
                        brand_data_clean = brand_monthly[brand].dropna()
                        if len(brand_data_clean) >= months:
                            start_value = brand_data_clean.iloc[-months]
                            if start_value > 0:  # Avoid division by zero
                                end_value = brand_data_clean.iloc[-1]
                                growth = ((end_value - start_value) / start_value) * 100
                                brand_growth[brand] = growth
                    
                    # Create and display growth chart
                    growth_data = pd.DataFrame(list(brand_growth.items()), columns=['Brand', 'Growth'])
                    growth_data = growth_data.sort_values('Growth', ascending=False)
                    
                    fig = px.bar(
                        growth_data,
                        x='Brand', 
                        y='Growth',
                        labels={"Growth": "Growth Rate (%)", "Brand": "Brand"},
                        title=f"Brand Growth Rate ({period})"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("<div class='sub-header'>Brand Comparison</div>", unsafe_allow_html=True)
                
                # Select brands to compare
                compare_brands = st.multiselect(
                    "Select Brands to Compare", 
                    options=brand_sales.index.tolist(),
                    default=brand_sales.index[:2].tolist()
                )
                
                if len(compare_brands) >= 2:
                    # Filter data for selected brands
                    brand_compare_df = df[df['Brand'].isin(compare_brands)]
                    
                    # Brand comparison metrics
                    brand_metrics = {}
                    for brand in compare_brands:
                        brand_data = brand_compare_df[brand_compare_df['Brand'] == brand]
                        metrics = {
                            'Total Sales': brand_data['Sales'].sum(),
                            'Avg Order Value': brand_data['Sales'].mean(),
                            'Total Quantity': brand_data['Quantity'].sum() if 'Quantity' in brand_data.columns else 0,
                            'Profit': brand_data['Profit'].sum() if 'Profit' in brand_data.columns else 0,
                            'Profit Margin': (brand_data['Profit'].sum() / brand_data['Sales'].sum() * 100) if 'Profit' in brand_data.columns else 0
                        }
                        brand_metrics[brand] = metrics
                    
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame(brand_metrics).T
                    
                    # Create radar chart data
                    radar_metrics = comparison_df.copy()
                    # Normalize metrics for radar chart
                    for col in radar_metrics.columns:
                        if radar_metrics[col].max() > 0:
                            radar_metrics[col] = radar_metrics[col] / radar_metrics[col].max() * 100
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    for brand in radar_metrics.index:
                        fig.add_trace(go.Scatterpolar(
                            r=radar_metrics.loc[brand].values,
                            theta=radar_metrics.columns,
                            fill='toself',
                            name=brand
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        title="Brand Comparison Radar Chart (Normalized)",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show metrics table
                    st.markdown("<div class='sub-header'>Brand Metrics Comparison</div>", unsafe_allow_html=True)
                    
                    # Format metrics
                    display_comparison = comparison_df.copy()
                    display_comparison['Total Sales'] = display_comparison['Total Sales'].map('₹{:,.2f}'.format)
                    display_comparison['Avg Order Value'] = display_comparison['Avg Order Value'].map('₹{:,.2f}'.format)
                    display_comparison['Total Quantity'] = display_comparison['Total Quantity'].map('{:,.0f}'.format)
                    display_comparison['Profit'] = display_comparison['Profit'].map('₹{:,.2f}'.format)
                    display_comparison['Profit Margin'] = display_comparison['Profit Margin'].map('{:,.2f}%'.format)
                    
                    st.table(display_comparison)
                    
                    # Add category distribution comparison
                    if 'Category' in df.columns:
                        st.markdown("<div class='sub-header'>Category Distribution by Brand</div>", unsafe_allow_html=True)
                        
                        # Get category distribution for each brand
                        brand_categories = {}
                        for brand in compare_brands:
                            brand_data = brand_compare_df[brand_compare_df['Brand'] == brand]
                            cat_dist = brand_data.groupby('Category')['Sales'].sum().sort_values(ascending=False)
                            # Convert to percentages
                            cat_dist = (cat_dist / cat_dist.sum()) * 100
                            brand_categories[brand] = cat_dist
                        
                        # Create comparison chart
                        cat_data = pd.DataFrame(brand_categories)
                        cat_data = cat_data.fillna(0)
                        
                        # Limit to top categories
                        top_cats = cat_data.mean(axis=1).sort_values(ascending=False).head(10).index
                        cat_data = cat_data.loc[top_cats]
                        
                        # Create bar chart
                        fig = go.Figure()
                        
                        for brand in cat_data.columns:
                            fig.add_trace(go.Bar(
                                x=cat_data.index,
                                y=cat_data[brand],
                                name=brand
                            ))
                        
                        fig.update_layout(
                            title="Category Distribution by Brand (% of Brand Sales)",
                            xaxis_title="Category",
                            yaxis_title="Percentage of Brand Sales",
                            barmode='group',
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least two brands to compare")
        else:
            st.info("Brand data not available in the dataset")
    
    # CUSTOMER INSIGHTS PAGE
    # Customer Insights Page
    elif page == "Customer Insights":
        st.markdown("<div class='main-header'>Customer Insights</div>", unsafe_allow_html=True)

        # Check if we have customer data
        has_customer_data = ecommerce_data is not None and 'Customer ID' in ecommerce_data.columns
    
        if has_customer_data:
            # Customer metrics
            customer_df = ecommerce_data.copy()
            total_customers = customer_df['Customer ID'].nunique()
            avg_customer_value = customer_df.groupby('Customer ID')['Sales'].sum().mean()
            repeat_customers = customer_df.groupby('Customer ID').size()
            repeat_rate = (repeat_customers[repeat_customers > 1].count() / total_customers) * 100
        
        # Display metrics
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                st.metric("Total Customers", f"{total_customers:,}")
        
            with col2:
                st.metric("Avg Customer Value", f"₹{avg_customer_value:,.2f}")
        
            with col3:
                st.metric("Repeat Purchase Rate", f"{repeat_rate:.2f}%")
        
            with col4:
                avg_order_count = repeat_customers.mean()
                st.metric("Avg Orders per Customer", f"{avg_order_count:.2f}")
        
        # Customer analysis tabs
            tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Purchase Frequency", "Cohort Analysis"])
        
            with tab1:
                st.markdown("<div class='sub-header'>Customer Segmentation</div>", unsafe_allow_html=True)
            
            # RFM Analysis
                st.markdown("#### RFM Analysis (Recency, Frequency, Monetary)")
            
            # Calculate RFM metrics
                max_order_date = customer_df['Order Date'].max()
                rfm = customer_df.groupby('Customer ID').agg({
                    'Order Date': lambda x: (max_order_date - x.max()).days,  # Recency
                    'Product_ID': 'count',  # Frequency
                    'Sales': 'sum'  # Monetary
                })
                rfm.columns = ['Recency', 'Frequency', 'Monetary']
            
            # Create RFM score (1-5 scale for each)
                rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
                rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
            
            # Calculate overall RFM score
                rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
            
            # Define customer segments
                def segment_customer(row):
                    if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
                        return 'Champions'
                    elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                        return 'Loyal Customers'
                    elif row['R_Score'] >= 3 and row['F_Score'] >= 1 and row['M_Score'] >= 2:
                        return 'Potential Loyalists'
                    elif row['R_Score'] >= 4 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
                        return 'New Customers'
                    elif row['R_Score'] < 3 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
                        return 'At Risk'
                    elif row['R_Score'] < 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
                        return 'Cannot Lose'
                    elif row['R_Score'] < 2 and row['F_Score'] < 2 and row['M_Score'] < 2:
                        return 'Lost'
                    else:
                        return 'Others'
            
                rfm['Segment'] = rfm.apply(segment_customer, axis=1)
            
            # Display segment distribution
                segment_counts = rfm['Segment'].value_counts()
            
            # Segment distribution chart
                fig = px.pie(
                    names=segment_counts.index, 
                    values=segment_counts.values,
                    title="Customer Segmentation",
                    color_discrete_sequence=px.colors.qualitative.G10,
                    hole=0.4
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Segment metrics
                segment_metrics = rfm.groupby('Segment').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'mean',
                }).reset_index()
            
                segment_metrics = segment_metrics.rename(columns={'Customer ID': 'Count'})
                segment_metrics['Recency'] = segment_metrics['Recency'].map('{:.1f} days'.format)
                segment_metrics['Frequency'] = segment_metrics['Frequency'].map('{:.1f} orders'.format)
                segment_metrics['Monetary'] = segment_metrics['Monetary'].map('₹{:,.2f}'.format)
            
                st.markdown("#### Segment Characteristics")
                st.table(segment_metrics)
            
            # Add RFM distribution visualization
                st.markdown("#### RFM Distribution")
            
            # 3D scatter plot of RFM values
                fig = px.scatter_3d(
                    rfm,
                    x='Recency',
                    y='Frequency',
                    z='Monetary',
                    color='Segment',
                    opacity=0.7,
                    title="RFM 3D Distribution"
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show 2D plots
                col1, col2 = st.columns(2)
            
                with col1:
                # Frequency vs Monetary
                    fig = px.scatter(
                        rfm,
                        x='Frequency',
                        y='Monetary',
                        color='Segment',
                        title="Frequency vs Monetary Value",
                        opacity=0.7
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
                with col2:
                    # Recency vs Monetary
                    fig = px.scatter(
                        rfm,
                        x='Recency',
                        y='Monetary',
                        color='Segment',
                        title="Recency vs Monetary Value",
                        opacity=0.7
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
            with tab2:
                st.markdown("<div class='sub-header'>Purchase Frequency Analysis</div>", unsafe_allow_html=True)
            
            # Calculate purchase frequency
                purchase_counts = customer_df.groupby('Customer ID').size().value_counts().sort_index()
            
            # Plot purchase frequency
                fig = px.bar(
                    x=purchase_counts.index,
                    y=purchase_counts.values,
                    labels={"x": "Number of Purchases", "y": "Number of Customers"},
                    title="Purchase Frequency Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate time between purchases
                customer_orders = customer_df.sort_values('Order Date').groupby('Customer ID')['Order Date'].apply(list)
            
            # Calculate days between purchases for each customer
                days_between = []
                for orders in customer_orders:
                    if len(orders) >= 2:
                        for i in range(1, len(orders)):
                            days_between.append((orders[i] - orders[i-1]).days)
            
            # Plot time between purchases
                if days_between:
                    days_df = pd.DataFrame({'Days': days_between})
                    days_df = days_df[days_df['Days'] <= 365]  # Filter outliers
                
                    fig = px.histogram(
                        days_df,
                        x='Days',
                        nbins=50,
                        labels={"Days": "Days Between Purchases", "count": "Frequency"},
                        title="Time Between Purchases Distribution"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Calculate statistics
                    avg_days = np.mean(days_df['Days'])
                    median_days = np.median(days_df['Days'])
                
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Days Between Purchases", f"{avg_days:.2f}")
                    with col2:
                        st.metric("Median Days Between Purchases", f"{median_days:.2f}")
            
            # Customer Purchase Timeline
                st.markdown("#### Customer Purchase Timeline")
            
            # Select customer examples to display
                top_customers = customer_df.groupby('Customer ID')['Sales'].sum().sort_values(ascending=False).head(100).index
                selected_customers = st.multiselect(
                    "Select Customers to View Purchase Timeline", 
                    options=top_customers.tolist(),  # Ensure options is a list
                    default=top_customers[:5].tolist()  # Ensure default is a list
                )
            
                if selected_customers:
                # Create timeline data
                    timeline_data = customer_df[customer_df['Customer ID'].isin(selected_customers)]
                
                # Create scatter plot
                    fig = px.scatter(
                        timeline_data,
                        x='Order Date',
                        y='Customer ID',
                        size='Sales',
                        color='Sales',
                        labels={"Order Date": "Purchase Date", "Customer ID": "Customer", "Sales": "Order Value (₹)"},
                        title="Customer Purchase Timeline"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
            with tab3:
                st.markdown("<div class='sub-header'>Customer Cohort Analysis</div>", unsafe_allow_html=True)
            
            # Create cohort analysis
                customer_first_purchase = customer_df.groupby('Customer ID')['Order Date'].min().reset_index()
                customer_first_purchase.columns = ['Customer ID', 'First Purchase Date']
            
                cohort_df = customer_df.merge(customer_first_purchase, on='Customer ID')
                cohort_df['Cohort'] = cohort_df['First Purchase Date'].dt.to_period('M')
                cohort_df['Period'] = (cohort_df['Order Date'].dt.to_period('M') - cohort_df['Cohort']).apply(lambda x: x.n)
            
                cohort_size = cohort_df.groupby('Cohort')['Customer ID'].nunique()
                cohort_data = cohort_df.groupby(['Cohort', 'Period'])['Customer ID'].nunique().reset_index()
                retention_table = cohort_data.pivot_table(index='Cohort', columns='Period', values='Customer ID')
            
                retention_rates = retention_table.copy()
                for i in range(retention_table.shape[1]):
                    retention_rates.iloc[:, i] = retention_table.iloc[:, i] / cohort_size * 100
            
                retention_rates = retention_rates.iloc[:, :12]  # First 12 months
            
                fig = px.imshow(
                    retention_rates.values,
                    labels=dict(x="Month", y="Cohort", color="Retention Rate (%)"),
                    x=[str(i) for i in retention_rates.columns],
                    y=retention_rates.index.astype(str),
                    color_continuous_scale="YlGnBu",
                    title="Customer Cohort Retention Analysis"
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
                avg_retention = retention_rates.mean()
            
                fig = px.line(
                    x=avg_retention.index,
                    y=avg_retention.values,
                    markers=True,
                    labels={"x": "Month", "y": "Retention Rate (%)"},
                    title="Average Retention Rate by Month"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Customer data not available. Please upload e-commerce data with customer information.")
    
    # INVENTORY MANAGEMENT PAGE
    elif page == "Inventory Management":
        st.markdown("<div class='main-header'>Inventory Management</div>", unsafe_allow_html=True)
        
        if 'Current_Stock_Level' in df.columns or 'Inventory' in df.columns:
            # Use the available inventory column
            inventory_col = 'Current_Stock_Level' if 'Current_Stock_Level' in df.columns else 'Inventory'
            
            # Inventory metrics
            total_inventory = df[inventory_col].sum()
            avg_inventory = df[inventory_col].mean()
            
            # Calculate stock-out items
            stockout_count = df[df[inventory_col] == 0]['Product'].nunique()
            stockout_percent = (stockout_count / df['Product'].nunique()) * 100
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Inventory", f"{total_inventory:,.0f}")
            
            with col2:
                st.metric("Average Stock Level", f"{avg_inventory:.1f}")
            
            with col3:
                st.metric("Stockout Items", f"{stockout_count}")
            
            with col4:
                st.metric("Stockout Percentage", f"{stockout_percent:.2f}%")
            
            # Inventory analysis tabs
            tab1, tab2, tab3 = st.tabs(["Inventory Levels", "Stock Analysis", "Reorder Recommendations"])
            
            with tab1:
                st.markdown("<div class='sub-header'>Current Inventory Levels</div>", unsafe_allow_html=True)
                
                # Calculate days of supply if we have sales data
                if 'Quantity' in df.columns:
                    # Calculate daily sales rate
                    product_daily_sales = df.groupby('Product')['Quantity'].sum() / 30  # Assuming 30 days
                    
                    # Calculate days of supply
                    product_inventory = df.groupby('Product')[inventory_col].mean()
                    product_days = product_inventory / product_daily_sales
                    
                    # Cap at 90 days for better visualization
                    product_days = product_days.clip(upper=90)
                    
                    # Create days of supply visualization
                    days_df = pd.DataFrame({
                        'Product': product_days.index,
                        'Days of Supply': product_days.values
                    }).sort_values('Days of Supply')
                    
                    # Create inventory level chart
                    fig = px.bar(
                        days_df.head(20),
                        x='Product',
                        y='Days of Supply',
                        labels={"Days of Supply": "Days of Supply", "Product": "Product"},
                        title="Days of Supply for Low Inventory Products"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Inventory distribution
                inventory_dist = df.groupby('Product')[inventory_col].mean().reset_index()
                inventory_dist = inventory_dist.sort_values(inventory_col, ascending=False)
                
                # Plot inventory distribution
                fig = px.histogram(
                    inventory_dist,
                    x=inventory_col,
                    nbins=30,
                    labels={inventory_col: "Stock Level", "count": "Number of Products"},
                    title="Inventory Distribution Across Products"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top and bottom inventory products
                st.markdown("#### Products with Highest and Lowest Inventory")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Highest Stock Levels")
                    high_inv = inventory_dist.head(10)
                    high_inv[inventory_col] = high_inv[inventory_col].map('{:,.0f}'.format)
                    st.table(high_inv)
                
                with col2:
                    st.markdown("##### Lowest Stock Levels")
                    low_inv = inventory_dist.tail(10)
                    low_inv = low_inv.sort_values(inventory_col)
                    low_inv[inventory_col] = low_inv[inventory_col].map('{:,.0f}'.format)
                    st.table(low_inv)
            
            with tab2:
                st.markdown("<div class='sub-header'>Stock Analysis</div>", unsafe_allow_html=True)
                
                # Stock categories
                if 'Quantity' in df.columns:
                    # Calculate ABC classification
                    st.markdown("#### ABC Inventory Analysis")
                    
                    # Calculate sales volume for each product
                    product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
                    product_sales_cum = product_sales.cumsum() / product_sales.sum()
                    
                    # Classify products
                    classification = []
                    for cum_pct in product_sales_cum:
                        if cum_pct <= 0.8:
                            classification.append('A')
                        elif cum_pct <= 0.95:
                            classification.append('B')
                        else:
                            classification.append('C')
                    
                    # Create classification dataframe
                    abc_df = pd.DataFrame({
                        'Product': product_sales.index,
                        'Sales': product_sales.values,
                        'Cumulative %': product_sales_cum.values * 100,
                        'Class': classification
                    })
                    
                    # Get inventory data
                    product_inventory = df.groupby('Product')[inventory_col].mean()
                    abc_df['Inventory'] = abc_df['Product'].map(lambda x: product_inventory.get(x, 0))
                    
                    # Calculate metrics by class
                    class_metrics = abc_df.groupby('Class').agg({
                        'Product': 'count',
                        'Sales': 'sum',
                        'Inventory': 'sum'
                    })
                    
                    class_metrics['% of Products'] = class_metrics['Product'] / class_metrics['Product'].sum() * 100
                    class_metrics['% of Sales'] = class_metrics['Sales'] / class_metrics['Sales'].sum() * 100
                    class_metrics['% of Inventory'] = class_metrics['Inventory'] / class_metrics['Inventory'].sum() * 100
                    
                    # Format table for display
                    display_metrics = class_metrics.copy()
                    display_metrics['Product'] = display_metrics['Product'].map('{:,.0f}'.format)
                    display_metrics['Sales'] = display_metrics['Sales'].map('₹{:,.2f}'.format)
                    display_metrics['Inventory'] = display_metrics['Inventory'].map('{:,.0f}'.format)
                    display_metrics['% of Products'] = display_metrics['% of Products'].map('{:.2f}%'.format)
                    display_metrics['% of Sales'] = display_metrics['% of Sales'].map('{:.2f}%'.format)
                    display_metrics['% of Inventory'] = display_metrics['% of Inventory'].map('{:.2f}%'.format)
                    
                    # Display class metrics
                    st.table(display_metrics)
                    
                    # Visualize ABC classification
                    fig = px.bar(
                        abc_df,
                        x='Product',
                        y='Sales',
                        color='Class',
                        labels={"Product": "Products", "Sales": "Sales (₹)", "Class": "ABC Class"},
                        title="ABC Classification of Products",
                        color_discrete_sequence=['#FF4D4D', '#FFA64D', '#5CD65C']
                    )
                    fig.update_layout(
                        height=500,
                        xaxis_visible=False,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate inventory metrics by category
                    st.markdown("#### Inventory Health by Category")
                    
                    if 'Category' in df.columns:
                        # Calculate inventory turnover by category
                        category_sales = df.groupby('Category')['Sales'].sum()
                        category_inventory = df.groupby('Category')[inventory_col].mean()
                        
                        # Calculate inventory turnover ratio (annualized)
                        category_turnover = category_sales / category_inventory * (365 / 30)  # Assuming 30 days of data
                        
                        # Calculate days of supply
                        category_days = 365 / category_turnover
                        
                        # Create category metrics dataframe
                        category_metrics = pd.DataFrame({
                            'Sales': category_sales,
                            'Inventory': category_inventory,
                            'Turnover Ratio': category_turnover,
                            'Days of Supply': category_days
                        })
                        
                        # Sort by turnover
                        category_metrics = category_metrics.sort_values('Turnover Ratio', ascending=False)
                        
                        # Plot inventory turnover
                        fig = px.bar(
                            category_metrics.reset_index(),
                            x='Category',
                            y='Turnover Ratio',
                            labels={"Category": "Category", "Turnover Ratio": "Inventory Turnover Ratio (Annualized)"},
                            title="Inventory Turnover by Category"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot days of supply
                        fig = px.bar(
                            category_metrics.reset_index(),
                            x='Category',
                            y='Days of Supply',
                            labels={"Category": "Category", "Days of Supply": "Days of Supply"},
                            title="Days of Supply by Category"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Stockout analysis
                st.markdown("#### Stockout Analysis")
                
                # Get stockout products
                stockout_products = df[df[inventory_col] == 0]['Product'].unique()
                
                if len(stockout_products) > 0:
                    # Get sales data for stockout products
                    if 'Quantity' in df.columns:
                        stockout_sales = df[df['Product'].isin(stockout_products)].groupby('Product')['Sales'].sum()
                        stockout_sales = stockout_sales.sort_values(ascending=False)
                        
                        # Create stockout impact chart
                        fig = px.bar(
                            x=stockout_sales.index[:20],
                            y=stockout_sales.values[:20],
                            labels={"x": "Product", "y": "Sales (₹)"},
                            title="Top 20 Stockout Products by Sales Impact"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display stockout products
                    stockout_df = pd.DataFrame({'Product': stockout_products})
                    st.write(f"Total of {len(stockout_products)} products currently out of stock")
                    st.dataframe(stockout_df, use_container_width=True)
                else:
                    st.info("No stockout products detected")
            
            with tab3:
                st.markdown("<div class='sub-header'>Reorder Recommendations</div>", unsafe_allow_html=True)
                
                if 'Quantity' in df.columns:
                    # Calculate reorder points and recommendations
                    st.markdown("#### Reorder Recommendations")
                    
                    # Calculate average daily demand
                    product_daily_demand = df.groupby('Product')['Quantity'].sum() / 30  # Assuming 30 days
                    
                    # Set safety stock level (e.g., 7 days)
                    safety_days = st.slider("Safety Stock (Days)", 1, 30, 7)
                    
                    # Set lead time
                    lead_time = st.slider("Average Lead Time (Days)", 1, 30, 14)
                    
                    # Calculate reorder point
                    reorder_point = (product_daily_demand * lead_time) + (product_daily_demand * safety_days)
                    
                    # Get current inventory levels
                    current_inventory = df.groupby('Product')[inventory_col].mean()
                    
                    # Calculate items to reorder
                    reorder_df = pd.DataFrame({
                        'Product': product_daily_demand.index,
                        'Daily Demand': product_daily_demand.values,
                        'Current Stock': current_inventory.values,
                        'Reorder Point': reorder_point.values,
                        'Days of Supply': current_inventory.values / product_daily_demand.values,
                        'Order Quantity': np.maximum(0, reorder_point.values - current_inventory.values)
                    })
                    
                    # Filter items that need reordering
                    reorder_df = reorder_df[reorder_df['Current Stock'] <= reorder_df['Reorder Point']]
                    reorder_df = reorder_df.sort_values('Days of Supply')
                    
                    # Cap days of supply for better visualization
                    reorder_df['Days of Supply'] = reorder_df['Days of Supply'].clip(upper=30)
                    
                    # Plot reorder recommendations
                    fig = px.bar(
                        reorder_df.head(20),
                        x='Product',
                        y='Days of Supply',
                        labels={"Days of Supply": "Days of Supply", "Product": "Product"},
                        title="Products to Reorder (Top 20 Urgent Items)"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display reorder table
                    st.markdown("#### Detailed Reorder List")
                    
                    # Format for display
                    display_reorder = reorder_df.copy()
                    display_reorder['Daily Demand'] = display_reorder['Daily Demand'].map('{:.2f}'.format)
                    display_reorder['Current Stock'] = display_reorder['Current Stock'].map('{:.0f}'.format)
                    display_reorder['Reorder Point'] = display_reorder['Reorder Point'].map('{:.0f}'.format)
                    display_reorder['Days of Supply'] = display_reorder['Days of Supply'].map('{:.1f}'.format)
                    display_reorder['Order Quantity'] = display_reorder['Order Quantity'].map('{:.0f}'.format)
                    
                    st.dataframe(display_reorder, use_container_width=True)
                    
                    # Download reorder list
                    csv = display_reorder.to_csv(index=False)
                    st.download_button(
                        label="Download Reorder List",
                        data=csv,
                        file_name="reorder_recommendations.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Quantity data not available for reorder recommendations")
        else:
            st.info("Inventory data not available in the dataset")
    
    # SALES PREDICTION PAGE
    elif page == "Sales Prediction":
        st.markdown("<div class='main-header'>Sales Prediction</div>", unsafe_allow_html=True)
        
        # Check if we have enough time-series data
        if df['Order Date'].min() < df['Order Date'].max() - pd.Timedelta(days=60):
            # Create time series of daily sales
            sales_ts = df.groupby(df['Order Date'])['Sales'].sum().reset_index()
            
            # Create monthly time series
            monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
            
            # Display forecast options
            forecast_type = st.radio("Forecast Period", ["Monthly", "Daily"])
            
            if forecast_type == "Monthly":
                time_series = monthly_sales
                forecast_periods = st.slider("Number of Months to Forecast", 1, 12, 3)
                
                # Set up prophet input dataframe
                prophet_df = pd.DataFrame({
                    'ds': time_series.index.to_timestamp(),
                    'y': time_series.values
                })
                
                # Plot historical data
                st.markdown("#### Historical Monthly Sales")
                
                fig = px.line(
                    prophet_df,
                    x='ds',
                    y='y',
                    labels={"ds": "Date", "y": "Sales (₹)"},
                    title="Historical Monthly Sales Data"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate forecast
                st.markdown("#### Sales Forecast")
                
                try:
                    with st.spinner("Generating forecast..."):
                        # Initialize and fit Prophet model
                        model = Prophet(
                            seasonality_mode='multiplicative',
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False
                        )
                        
                        # Add country holidays if country data available
                        if 'Country' in df.columns:
                            country = df['Country'].mode()[0]
                            if country in ['US', 'USA', 'United States']:
                                model.add_country_holidays(country_name='US')
                        
                        # Fit the model
                        model.fit(prophet_df)
                        
                        # Create future dataframe
                        future = model.make_future_dataframe(periods=forecast_periods, freq='M')
                        
                        # Generate forecast
                        forecast = model.predict(future)
                        
                        # Create forecast plot
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=prophet_df['ds'],
                            y=prophet_df['y'],
                            mode='lines+markers',
                            name='Historical Sales',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast line
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'][-forecast_periods:],
                            y=forecast['yhat'][-forecast_periods:],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Add uncertainty intervals
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast['ds'][-forecast_periods:], forecast['ds'][-forecast_periods:].iloc[::-1]]),
                            y=pd.concat([forecast['yhat_lower'][-forecast_periods:], forecast['yhat_upper'][-forecast_periods:].iloc[::-1]]),
                            fill='toself',
                            fillcolor='rgba(231, 234, 241, 0.5)',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            hoverinfo='skip',
                            showlegend=False
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title='Monthly Sales Forecast',
                            xaxis_title='Date',
                            yaxis_title='Sales (₹)',
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast components
                        st.markdown("#### Forecast Components")
                        fig_comp = model.plot_components(forecast)
                        st.pyplot(fig_comp)
                        
                        # Display forecast table
                        st.markdown("#### Forecast Data")
                        
                        # Create forecast table
                        forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-forecast_periods:]
                        forecast_table.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                        
                        # Format for display
                        forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m')
                        forecast_table['Forecast'] = forecast_table['Forecast'].map('₹{:,.2f}'.format)
                        forecast_table['Lower Bound'] = forecast_table['Lower Bound'].map('₹{:,.2f}'.format)
                        forecast_table['Upper Bound'] = forecast_table['Upper Bound'].map('₹{:,.2f}'.format)
                        
                        st.table(forecast_table)
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    st.info("Try with different forecast parameters or check your data")
            
            else:  # Daily forecast
                # Aggregate to daily
                daily_sales = df.groupby(df['Order Date'])['Sales'].sum().reset_index()
                
                # Set up prophet input dataframe
                prophet_df = pd.DataFrame({
                    'ds': daily_sales['Order Date'],
                    'y': daily_sales['Sales']
                })
                
                # Forecast parameters
                forecast_days = st.slider("Number of Days to Forecast", 7, 90, 30)
                
                # Plot historical data
                st.markdown("#### Historical Daily Sales")
                
                fig = px.line(
                    prophet_df,
                    x='ds',
                    y='y',
                    labels={"ds": "Date", "y": "Sales (₹)"},
                    title="Historical Daily Sales Data"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate forecast
                st.markdown("#### Sales Forecast")
                
                try:
                    with st.spinner("Generating forecast..."):
                        # Initialize and fit Prophet model
                        model = Prophet(
                            seasonality_mode='multiplicative',
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            changepoint_prior_scale=0.05
                        )
                        
                        # Add country holidays if country data available
                        if 'Country' in df.columns:
                            country = df['Country'].mode()[0]
                            if country in ['US', 'USA', 'United States']:
                                model.add_country_holidays(country_name='US')
                        
                        # Fit the model
                        model.fit(prophet_df)
                        
                        # Create future dataframe
                        future = model.make_future_dataframe(periods=forecast_days)
                        
                        # Generate forecast
                        forecast = model.predict(future)
                        
                        # Create forecast plot
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=prophet_df['ds'],
                            y=prophet_df['y'],
                            mode='lines',
                            name='Historical Sales',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast line
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'][-forecast_days:],
                            y=forecast['yhat'][-forecast_days:],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Add uncertainty intervals
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast['ds'][-forecast_days:], forecast['ds'][-forecast_days:].iloc[::-1]]),
                            y=pd.concat([forecast['yhat_lower'][-forecast_days:], forecast['yhat_upper'][-forecast_days:].iloc[::-1]]),
                            fill='toself',
                            fillcolor='rgba(231, 234, 241, 0.5)',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            hoverinfo='skip',
                            showlegend=False
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title='Daily Sales Forecast',
                            xaxis_title='Date',
                            yaxis_title='Sales (₹)',
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast components
                        st.markdown("#### Forecast Components")
                        fig_comp = model.plot_components(forecast)
                        st.pyplot(fig_comp)
                        
                        # Display forecast summary
                        st.markdown("#### Forecast Summary")
                        
                        # Calculate monthly totals from daily forecast
                        forecast['month'] = forecast['ds'].dt.to_period('M')
                        monthly_summary = forecast.groupby('month').agg({
                            'yhat': 'sum',
                            'yhat_lower': 'sum',
                            'yhat_upper': 'sum'
                        }).reset_index()
                        
                        monthly_summary['month'] = monthly_summary['month'].astype(str)
                        
                        # Plot monthly summary
                        fig = px.bar(
                            monthly_summary,
                            x='month',
                            y='yhat',
                            error_y=monthly_summary['yhat_upper'] - monthly_summary['yhat'],
                            labels={'month': 'Month', 'yhat': 'Forecasted Sales (₹)'},
                            title='Monthly Sales Forecast Summary'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    st.info("Try with different forecast parameters or check your data")
        else:
            st.warning("Insufficient time-series data for forecasting. Need at least 60 days of data.")
    
    # SUSTAINABILITY PAGE
    elif page == "Sustainability":
        st.markdown("<div class='main-header'>Sustainability Dashboard</div>", unsafe_allow_html=True)
    
    # Check if the required columns are present
        if 'Sustainability_Score' in df.columns and 'Carbon_Emissions_Per_Order' in df.columns and 'Sustainable_Packaging_Flag' in df.columns:
            # Visualize Sustainability_Score vs. Carbon_Emissions_Per_Order
            st.markdown("<div class='sub-header'>Sustainability Score vs. Carbon Emissions</div>", unsafe_allow_html=True)
        
            fig = px.scatter(
                df,
                x='Sustainability_Score',
                y='Carbon_Emissions_Per_Order',
                color='Sustainable_Packaging_Flag',
                labels={"Sustainability_Score": "Sustainability Score", "Carbon_Emissions_Per_Order": "Carbon Emissions per Order", "Sustainable_Packaging_Flag": "Sustainable Packaging"},
                title="Sustainability Score vs. Carbon Emissions per Order"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Visualize Sustainable_Packaging_Flag vs. Carbon_Emissions_Per_Order
            st.markdown("<div class='sub-header'>Sustainable Packaging vs. Carbon Emissions</div>", unsafe_allow_html=True)
        
            fig = px.box(
                df,
                x='Sustainable_Packaging_Flag',
                y='Carbon_Emissions_Per_Order',
                labels={"Sustainable_Packaging_Flag": "Sustainable Packaging", "Carbon_Emissions_Per_Order": "Carbon Emissions per Order"},
                title="Sustainable Packaging vs. Carbon Emissions per Order"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Required columns for sustainability analysis are missing. Please ensure 'Sustainability_Score', 'Carbon_Emissions_Per_Order', and 'Sustainable_Packaging_Flag' are present in your dataset.")

# Check if this is the first run
if 'first_run' not in st.session_state:
    st.session_state['first_run'] = True
    
    # Display welcome message
    st.balloons()
    st.success("Welcome to the Sales Analytics Dashboard!")
    
    # Set default settings
    if 'theme' not in st.session_state:
        st.session_state['theme'] = "Light"
        st.session_state['primary_color'] = "#1E88E5"
        st.session_state['background_color'] = "#FFFFFF"
        st.session_state['text_color'] = "#333333"
    
    if 'data_cache_ttl' not in st.session_state:
        st.session_state['data_cache_ttl'] = 60
    
    if 'notification_threshold' not in st.session_state:
        st.session_state['notification_threshold'] = 7
    
    if 'enable_notifications' not in st.session_state:
        st.session_state['enable_notifications'] = True