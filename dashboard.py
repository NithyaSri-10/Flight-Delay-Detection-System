import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# -------------------------------------------------------
# 🛠 PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="Flight Delay Prediction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# 🔒 SESSION STATE (Authentication)
# -------------------------------------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

API_BASE_URL = "http://localhost:5000"

# -------------------------------------------------------
# 🧠 UTILS
# -------------------------------------------------------
def safe_int(value):
    """Safely convert any input to int; return 0 if invalid"""
    try:
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().replace('.', '', 1).isdigit():
            return int(float(value))
        return 0
    except Exception:
        return 0


def get_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


@st.cache_data
def fetch_json(endpoint):
    """Fetch JSON safely from backend API"""
    try:
        res = requests.get(f"{API_BASE_URL}/{endpoint}", headers=get_headers(), timeout=10)
        if res.status_code == 200:
            return res.json()
        return {}
    except Exception as e:
        st.error(f"❌ API Error ({endpoint}): {e}")
        return {}

# -------------------------------------------------------
# 🎨 CSS STYLING
# -------------------------------------------------------
st.markdown("""
    <style>
    .header-title {
        color: #1f77b4;
        font-size: 2.3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🧩 LOGIN / REGISTER PAGE
# -------------------------------------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## ✈️ Flight Delay Prediction System")
        st.markdown("---")

        mode = st.radio("Choose Action", ["Login", "Register"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if mode == "Register":
            full_name = st.text_input("Full Name")
            confirm = st.text_input("Confirm Password", type="password")
        else:
            full_name = ""
            confirm = password

        if st.button(mode, type="primary"):
            try:
                if mode == "Login":
                    res = requests.post(f"{API_BASE_URL}/api/auth/login", json={"email": email, "password": password})
                else:
                    if password != confirm:
                        st.error("Passwords do not match.")
                        return
                    res = requests.post(
                        f"{API_BASE_URL}/api/auth/register",
                        json={"email": email, "password": password, "full_name": full_name}
                    )

                if res.status_code in [200, 201]:
                    data = res.json()
                    st.session_state.authenticated = True
                    st.session_state.token = data['token']
                    st.session_state.user = data['user']
                    st.success(f"Welcome, {data['user']['full_name'] or data['user']['email']}!")
                    st.rerun()
                else:
                    st.error(res.json().get('error', 'Authentication failed'))
            except Exception as e:
                st.error(str(e))

# -------------------------------------------------------
# MAIN APP
# -------------------------------------------------------
if not st.session_state.authenticated:
    login_page()
else:
    with st.sidebar:
        st.write(f"**User:** {st.session_state.user['full_name'] or st.session_state.user['email']}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Predictions", "Analytics", "Model Performance", "Clustering Analysis", "Weather Impact"]
    )

    # -------------------------------------------------------
    # 📊 DASHBOARD
    # -------------------------------------------------------
    if page == "Dashboard":
        st.markdown('<div class="header-title">📊 Flight Delay Analytics Dashboard</div>', unsafe_allow_html=True)

        stats = fetch_json("api/analytics/overall") or {}

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flights", f"{safe_int(stats.get('total_flights')):,}")
        col2.metric("Delayed Flights", f"{safe_int(stats.get('delayed_flights')):,}")
        col3.metric("Delay Rate", f"{stats.get('delay_percentage', 0):.1f}%")
        col4.metric("Avg Arrival Delay", f"{stats.get('avg_arrival_delay', 0):.1f} min")

        st.divider()

        # Hourly delay trends
        st.subheader("⏰ Delay by Hour of Day")
        hourly = fetch_json("api/analytics/by-hour")
        if isinstance(hourly, list) and hourly:
            df = pd.DataFrame(hourly)
            fig = px.line(df, x='hour', y='delay_percentage', markers=True,
            title="Delay Percentage by Hour", color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data available.")

        st.divider()

        # Day of week
        st.subheader("📅 Delay by Day of Week")
        daily = fetch_json("api/analytics/by-day")
        if isinstance(daily, list) and daily:
            df = pd.DataFrame(daily)
            fig = px.bar(df, x='day_name', y='delay_percentage', color='delay_percentage',
            color_continuous_scale='Reds', title="Delay Percentage by Day")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily data available.")

        st.divider()

        # Monthly delay
        st.subheader("🗓️ Delay by Month")
        monthly = fetch_json("api/analytics/by-month")
        if isinstance(monthly, list) and monthly:
            df = pd.DataFrame(monthly)
            fig = px.bar(df, x='month_name', y='delay_percentage', color='delay_percentage',
            color_continuous_scale='Blues', title="Delay Percentage by Month")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly data available.")

    # -------------------------------------------------------
    # 🔮 PREDICTIONS (AIRLINE-STYLE DASHBOARD VIEW)
    # -------------------------------------------------------
    elif page == "Predictions":
        st.markdown('<div class="header-title">🔮 Flight Delay Prediction</div>', unsafe_allow_html=True)

        # --- Input Layout ---
        col1, col2 = st.columns(2)
        with col1:
            airline = st.selectbox("✈️ Airline", ["AA", "DL", "UA", "SW", "B6", "AS"])
            origin = st.selectbox("🛫 Origin Airport", ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MIA"])
            distance = st.number_input("📏 Distance (miles)", min_value=0, max_value=3000, value=500)
        with col2:
            dest = st.selectbox("🛬 Destination Airport", ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MIA"])
            flight_date = st.date_input("📅 Flight Date")
            crs_dep_time = st.time_input("⏰ Scheduled Departure Time")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Predict Button ---
        if st.button("🚀 Predict Delay", type="primary"):
            payload = {
                'airline': airline,
                'origin': origin,
                'dest': dest,
                'distance': distance,
                'flight_date': flight_date.strftime('%Y-%m-%d'),
                'crs_dep_time': int(crs_dep_time.strftime('%H%M'))
            }

            try:
                res = requests.post(f"{API_BASE_URL}/api/predict", json=payload, headers=get_headers())

                if res.status_code == 200:
                    result = res.json()
                    classification = result.get("classification", {})
                    regression = result.get("regression", {})

                    st.divider()
                    st.markdown("### 🧠 Prediction Results")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # --- Two-column layout ---
                    left, right = st.columns(2)

                    # --- LEFT COLUMN: Classification ---
                    with left:
                        st.markdown("#### 🎯 Classification")
                        prediction = classification.get("prediction", "Unknown")
                        delay_prob = classification.get("delay_probability", 0)
                        ontime_prob = classification.get("on_time_probability", 0)

                        # Icon color for status
                        color_icon = "🔴" if prediction == "Delayed" else "🟢"
                        st.markdown(f"##### {color_icon} Prediction: **{prediction}**")
                        st.metric("Delay Probability", f"{delay_prob*100:.1f}%")
                        st.metric("On-Time Probability", f"{ontime_prob*100:.1f}%")

                    # --- RIGHT COLUMN: Regression + Gauge ---
                    with right:
                        st.markdown("#### 📈 Regression")
                        predicted_delay = regression.get("predicted_delay_minutes", 0.0)
                        st.metric("Predicted Delay", f"{predicted_delay:.1f} minutes")

                        # Gauge visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=predicted_delay,
                            title={'text': "Predicted Delay (minutes)"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 120]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 15], 'color': "lightgreen"},
                                    {'range': [15, 60], 'color': "lightyellow"},
                                    {'range': [60, 120], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 15
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(res.json().get("error", "Prediction failed. Please try again."))

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # -------------------------------------------------------
    # 📊 ANALYTICS (FULLY FIXED)
    # -------------------------------------------------------
    elif page == "Analytics":
        st.markdown('<div class="header-title">📊 Detailed Analytics</div>', unsafe_allow_html=True)

        # Fetch all data safely
        airline_stats = fetch_json("api/analytics/by-airline") or []
        route_stats = fetch_json("api/analytics/by-route?limit=20") or []
        hour_stats = fetch_json("api/analytics/by-hour") or []
        day_stats = fetch_json("api/analytics/by-day") or []
        month_stats = fetch_json("api/analytics/by-month") or []

        # Tabs for analytics
        tab1, tab2, tab3 = st.tabs(["By Airline", "By Route", "Drill-Down"])

        # ---------------------------------------------------
        # ✈️ By Airline
        # ---------------------------------------------------
        with tab1:
            st.subheader("Delay Statistics by Airline")

            if airline_stats:
                df_airline = pd.DataFrame(airline_stats)
                # Normalize key names
                df_airline.columns = [c.lower() for c in df_airline.columns]

                # Determine which columns exist
                y_col = "delay_percentage" if "delay_percentage" in df_airline.columns else "avg_arrival_delay"

                st.dataframe(df_airline, use_container_width=True)

                fig = px.bar(
                    df_airline,
                    x=df_airline.columns[0],
                    y=y_col,
                    color=y_col,
                    color_continuous_scale="Reds",
                    title="Delay Rate / Avg Delay by Airline",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No airline statistics available yet.")

        # ---------------------------------------------------
        # 🌍 By Route
        # ---------------------------------------------------
        with tab2:
            st.subheader("Delay Statistics by Route")

            if route_stats:
                df_routes = pd.DataFrame(route_stats)
                df_routes.columns = [c.lower() for c in df_routes.columns]
                st.dataframe(df_routes, use_container_width=True)

                y_col = "delay_percentage" if "delay_percentage" in df_routes.columns else "avg_arrival_delay"

                fig = px.scatter(
                    df_routes,
                    x="total_flights" if "total_flights" in df_routes.columns else df_routes.columns[0],
                    y=y_col,
                    size="delayed_flights" if "delayed_flights" in df_routes.columns else None,
                    hover_data=[df_routes.columns[0]],
                    color=y_col,
                    color_continuous_scale="Viridis",
                    title="Route Volume vs Delay Rate",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No route statistics available yet.")

        # ---------------------------------------------------
        # 🔍 Drill-down
        # ---------------------------------------------------
        with tab3:
            st.subheader("Drill-down Analysis (Year → Month → Day)")

            if not route_stats:
                st.warning("No route data available for drill-down.")
            else:
                routes = [r.get("route_code", f"Route-{i}") for i, r in enumerate(route_stats)]
                route_code = st.selectbox("Select Route", routes)

                if route_code:
                    try:
                        drill_data = fetch_json(f"api/analytics/drill-down/{route_code}") or []
                        if drill_data:
                            df_drill = pd.DataFrame(drill_data)
                            df_drill.columns = [c.lower() for c in df_drill.columns]

                            st.dataframe(df_drill, use_container_width=True)

                            y_col = "delay_percentage" if "delay_percentage" in df_drill.columns else "avg_arrival_delay"

                            fig = px.line(
                                df_drill,
                                x="month" if "month" in df_drill.columns else df_drill.columns[0],
                                y=y_col,
                                markers=True,
                                title=f"Delay Trend for Route {route_code}",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No drill-down data found for this route.")
                    except Exception as e:
                        st.error(f"Error fetching drill-down: {str(e)}")

    # -------------------------------------------------------
    # 🤖 MODEL PERFORMANCE
    # -------------------------------------------------------
    elif page == "Model Performance":
        st.markdown('<div class="header-title">🤖 Model Performance</div>', unsafe_allow_html=True)
        metrics = fetch_json("api/model-metrics")
        if metrics and 'classification' in metrics:
            df = pd.DataFrame(metrics['classification']).T.reset_index()
            df.rename(columns={'index': 'Model'}, inplace=True)
            st.dataframe(df)
            fig = px.bar(df, x='Model', y='accuracy', color='accuracy',
            color_continuous_scale='Blues', title="Model Accuracy Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance metrics available.")
    # -------------------------------------------------------
    # 🎯 CLUSTERING ANALYSIS (ENHANCED)
    # -------------------------------------------------------
    elif page == "Clustering Analysis":
        st.markdown('<div class="header-title">🎯 Airport Clustering Analysis</div>', unsafe_allow_html=True)

        clusters = fetch_json("api/clustering/airport-clusters")
        if clusters and 'clusters' in clusters:
            n_clusters = clusters.get('n_clusters', len(clusters['clusters']))
            st.subheader(f"Total Clusters: {n_clusters}")

            # Display each cluster with airports inside an expander
            for cluster_id, airports in clusters['clusters'].items():
                with st.expander(f"Cluster {cluster_id} ({len(airports)} airports)"):
                    st.write(", ".join(airports))

            st.divider()

            # Visualization: bar chart of airport counts
            cluster_data = [
                {'Cluster': f'Cluster {cid}', 'Airports': len(airports)}
                for cid, airports in clusters['clusters'].items()
            ]
            df_clusters = pd.DataFrame(cluster_data)

            fig = px.bar(
                df_clusters,
                x='Cluster',
                y='Airports',
                color='Airports',
                color_continuous_scale='Blues',
                title='Number of Airports in Each Cluster'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No clustering data available. Please train or run clustering first.")

    # -------------------------------------------------------
    # 🌦️ WEATHER IMPACT (CARD STYLE + SAFE)
    # -------------------------------------------------------
    elif page == "Weather Impact":
        st.markdown('<div class="header-title">🌦️ Weather Impact Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Origin Airport", ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MIA"])
        with col2:
            destination = st.selectbox("Destination Airport", ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MIA"])

        if st.button("Analyze Weather Impact", type="primary"):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/api/weather/impact",
                    json={"origin": origin, "destination": destination},
                    headers=get_headers()
                )

                if res.status_code == 200:
                    data = res.json()
                    st.divider()
                    st.markdown("### ✈️ Current Weather Conditions")

                    col1, col2 = st.columns(2)
                    for col, loc in zip([col1, col2], ['origin', 'destination']):
                        section = data.get(loc, {})
                        weather = section.get('weather')
                        impact = section.get('impact', {'risk_factor': 0})

                        with col:
                            st.markdown(f"#### {loc.capitalize()} ({section.get('airport', 'Unknown')})")

                            if weather:
                                st.markdown(f"""
                                    <div style='background-color:#f8f9fa;padding:15px;border-radius:10px;border:1px solid #ddd;'>
                                    🌡️ <b>Temperature:</b> {weather.get('temperature', 0)}°C<br>
                                    💨 <b>Wind Speed:</b> {weather.get('wind_speed', 0)} m/s<br>
                                    ☁️ <b>Condition:</b> {weather.get('conditions', 'N/A')}<br>
                                    ⚠️ <b>Risk Factor:</b> {impact.get('risk_factor', 0)*100:.1f}%
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div style='background-color:#fff3cd;padding:15px;border-radius:10px;border:1px solid #ffeeba;'>
                                    ⚠️ Weather data not available.
                                    </div>
                                """, unsafe_allow_html=True)

                    st.divider()
                    combined_risk = data.get('combined_risk', 0)

                    # Gauge chart for combined risk
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=combined_risk * 100,
                        title={'text': "Combined Weather Risk (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 20], 'color': 'lightgreen'},
                                {'range': [20, 60], 'color': 'yellow'},
                                {'range': [60, 100], 'color': 'red'}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': combined_risk * 100}
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(res.json().get('error', 'Weather analysis failed'))

            except Exception as e:
                st.error(f"Error: {str(e)}")
