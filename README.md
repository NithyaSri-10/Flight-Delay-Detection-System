# Flight Delay Prediction System

A comprehensive machine learning system for predicting flight delays using historical US airline data, built with a Data Warehouse, OLAP analytics, and multiple predictive models.

## Project Overview

This 8-week project implements a complete flight delay prediction system following industry best practices:

- **Week 1**: Problem definition and project planning
- **Week 2**: Data collection and warehouse schema design
- **Week 3**: ETL pipeline and data loading
- **Week 4**: OLAP queries and analytics engine
- **Week 5**: Classification models (Random Forest, Logistic Regression)
- **Week 6**: Regression and clustering models
- **Week 7**: System integration and dashboard
- **Week 8**: Testing, documentation, and final report

## Architecture

### Components

1. **Data Warehouse** (SQLAlchemy + SQLite/PostgreSQL)
   - Fact table: Flight records with delay metrics
   - Dimension tables: Airlines, Airports, Time, Routes

2. **ETL Pipeline**
   - Extract: Kaggle US Airline Flight Delay dataset
   - Transform: Data cleaning, feature engineering
   - Load: Populate warehouse tables

3. **ML Models**
   - Classification: Random Forest, Logistic Regression
   - Regression: Gradient Boosting (predict exact delay)
   - Clustering: K-Means (airport pattern analysis)

4. **Analytics Engine** (OLAP)
   - Delay statistics by airline, airport, route, hour, day, month
   - Drill-down analysis for detailed insights
   - Feature importance analysis

5. **APIs** (Flask)
   - Prediction endpoints
   - Analytics endpoints
   - Model metrics endpoints
   - Clustering analysis endpoints

6. **Dashboard** (Streamlit)
   - Interactive visualizations
   - Real-time predictions
   - Comprehensive analytics
   - Model performance metrics

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository
\`\`\`bash
git clone <repository-url>
cd flight-delay-predictor
\`\`\`

2. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Download dataset
\`\`\`bash
python scripts/download_kaggle_data.py
\`\`\`

4. Train models
\`\`\`bash
python scripts/train_all_models.py
\`\`\`

## Usage

### Run Flask API
\`\`\`bash
python app.py
\`\`\`
API will be available at `http://localhost:5000`

### Run Streamlit Dashboard
\`\`\`bash
streamlit run dashboard.py
\`\`\`
Dashboard will be available at `http://localhost:8501`

### Run Tests
\`\`\`bash
pytest tests/
\`\`\`

## API Endpoints

### Prediction
- `POST /api/predict` - Predict flight delay

## Dashboard Features

1. **Dashboard Tab**
   - Overall statistics
   - Delay patterns by hour, day, month
   - Top delayed routes

2. **Predictions Tab**
   - Interactive flight delay prediction
   - Classification and regression results
   - Probability gauges

3. **Analytics Tab**
   - Airline statistics
   - Route analysis
   - Drill-down capabilities

4. **Model Performance Tab**
   - Classification metrics
   - Regression metrics
   - Feature importance

5. **Clustering Analysis Tab**
   - Airport clusters
   - Cluster characteristics

## Model Performance

### Classification Models
- **Random Forest**: Accuracy ~85%, F1-Score ~0.82
- **Logistic Regression**: Accuracy ~78%, F1-Score ~0.75

### Regression Model
- **Gradient Boosting**: RMSE ~18 minutes, R² ~0.72

### Clustering
- **K-Means**: 5 airport clusters based on delay patterns

## Data Schema

### Fact Table (fact_flight)
- flight_id (PK)
- airline_id (FK)
- origin_airport_id (FK)
- dest_airport_id (FK)
- route_id (FK)
- time_id (FK)
- departure_delay, arrival_delay
- is_delayed, is_cancelled, is_diverted
- distance, air_time

### Dimension Tables
- **dim_airline**: airline_code, airline_name
- **dim_airport**: airport_code, airport_name, city, state
- **dim_time**: flight_date, year, month, day, day_of_week, quarter, week_of_year, is_weekend
- **dim_route**: origin_airport_id, dest_airport_id, route_code

## Key Insights

1. **Temporal Patterns**: Delays increase during peak hours (morning and evening)
2. **Seasonal Trends**: Summer months show higher delay rates
3. **Route Analysis**: Certain routes consistently experience delays
4. **Airline Performance**: Significant variation in delay rates across airlines
5. **Feature Importance**: Departure hour, airline, and route are top predictors

## Future Enhancements

- Real-time data integration
- Weather data incorporation
- Advanced ensemble models
- Mobile app development
- Predictive alerts system
- Cost-benefit analysis for airlines

## Team

- Data Engineering: ETL pipeline, warehouse design
- ML Engineering: Model training, feature engineering
- Analytics: OLAP queries, insights generation
- Full-Stack: API development, dashboard creation

## License

MIT License

## Contact

For questions or support, please contact the development team.
