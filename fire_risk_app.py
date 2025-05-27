# fire_risk_app.py
import warnings
import sys
import os

# Completely suppress Streamlit warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Verify Streamlit context
if not get_script_run_ctx():
    from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx
    import threading
    add_script_run_ctx(threading.current_thread())

# Rest of your imports...
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
from io import StringIO
from datetime import datetime, timedelta

# Country bounding boxes for automatic coordinate detection (extended)
country_bounding_boxes = {
    "afghanistan": "60.5,29.4,77.3,38.5",
    "albania": "19.3,39.6,21.1,42.7",
    "algeria": "-8.7,19.3,11.3,37.1",
    "andorra": "1.3,42.4,1.7,42.6",
    "angola": "11.5,-18.0,24.4,-4.4",
    "antigua and barbuda": "-61.8,17.0,-61.7,17.1",
    "argentina": "-73.6,-55.0,-53.7,-21.8",
    "armenia": "40.1,38.8,46.7,41.3",
    "australia": "112.92,-43.74,153.64,-10.69",
    "austria": "9.5,47.3,17.2,49.0",
    "azerbaijan": "44.5,38.4,50.4,41.6",
    "bahamas": "-78.0,23.5,-75.0,25.5",
    "bahrain": "50.3,25.4,50.6,26.2",
    "bangladesh": "88.0,20.4,92.7,26.6",
    "barbados": "-59.5,12.9,-59.3,13.3",
    "belarus": "23.1,51.3,32.0,56.0",
    "belgium": "2.5,49.5,6.4,51.5",
    "belize": "-89.2,15.8,-88.2,18.0",
    "benin": "0.7,6.2,3.1,12.5",
    "bhutan": "88.5,26.6,92.2,28.3",
    "bolivia": "-69.3,-22.0,-57.5,-9.7",
    "bosnia and herzegovina": "15.7,42.0,19.6,44.7",
    "botswana": "20.0,-26.4,29.0,-19.6",
    "brazil": "-73.99,-33.75,-34.79,5.27",
    "brunei": "113.0,4.2,114.5,5.5",
    "bulgaria": "22.4,41.3,28.5,44.4",
    "burkina faso": "-5.5,9.4,-1.5,15.0",
    "burundi": "29.9,-4.5,30.9,-2.9",
    "cabo verde": "-25.0,14.7,-22.6,16.3",
    "cambodia": "102.3,10.4,107.6,14.7",
    "cameroon": "9.2,2.5,16.5,13.4",
    "canada": "-141.0,41.7,-52.6,83.1",
    "central african republic": "14.2,2.2,27.3,11.1",
    "chad": "12.0,7.2,23.0,23.5",
    "chile": "-75.7,-56.0,-66.4,-17.5",
    "china": "73.5,18.2,135.0,53.6",
    "colombia": "-79.0,-4.2,-66.8,13.4",
    "comoros": "43.1,-12.2,44.6,-11.0",
    "congo (Congo-Brazzaville)": "12.1,-5.0,18.5,5.0",
    "congo (Congo-Kinshasa)": "12.0,-13.5,31.3,5.0",
    "costa rica": "-85.5,8.0,-82.5,11.0",
    "croatia": "13.0,42.5,19.5,45.1",
    "cuba": "-85.0,19.5,-74.0,23.0",
    "cyprus": "32.0,34.5,34.0,35.5",
    "czech republic": "12.0,48.5,18.9,51.1",
    "denmark": "8.0,54.5,15.0,57.5",
    "djibouti": "42.5,10.5,43.5,12.5",
    "dominica": "-61.5,15.2,-61.1,15.6",
    "dominican republic": "-71.5,17.5,-68.3,19.9",
    "east timor": "123.0,-9.5,127.0,-8.5",
    "ecuador": "-92.0,-5.0,-75.0,2.0",
    "egypt": "24.7,22.0,36.9,31.7",
    "el salvador": "-90.2,13.3,-87.6,14.4",
    "equatorial guinea": "5.5,1.5,11.5,3.5",
    "eritrea": "36.4,12.4,43.0,18.0",
    "estonia": "21.8,59.2,28.2,59.9",
    "eswatini": "30.5,-27.5,32.0,-26.0",
    "ethiopia": "33.0,3.5,48.0,15.0",
    "fiji": "177.5,-20.0,179.5,-17.0",
    "finland": "20.5,60.0,31.6,70.1",
    "france": "-5.1,41.3,9.6,51.1",
    "gabon": "9.0,-3.0,14.5,3.0",
    "gambia": "-16.4,13.4,-13.5,13.8",
    "georgia": "41.0,41.0,46.7,43.6",
    "germany": "5.9,47.3,15.0,55.1",
    "ghana": "-3.3,4.7,1.5,11.7",
    "greece": "19.3,35.8,28.2,42.1",
    "grenada": "-61.9,12.0,-61.3,12.5",
    "guatemala": "-92.2,13.0,-88.0,15.0",
    "guinea": "-15.0,7.0,-7.0,12.5",
    "guinea-bissau": "-15.0,11.0,-13.5,12.5",
    "guyana": "-61.5,1.5,-56.0,6.5",
    "haiti": "-74.5,18.0,-71.5,20.0",
    "honduras": "-89.3,13.0,-83.0,15.0",
    "hungary": "16.0,45.8,22.0,48.5",
    "iceland": "-24.0,63.5,-13.0,66.5",
    "india": "68.7,6.7,97.25,35.5",
    "indonesia": "95.0,-10.4,141.0,5.9",
    "iran": "44.0,24.8,63.3,39.8",
    "iraq": "38.8,29.1,48.0,37.4",
    "ireland": "-10.5,51.5,-5.5,55.5",
    "israel": "34.3,29.5,35.7,33.5",
    "italy": "6.6,36.6,18.5,47.1",
    "ivory coast": "-7.5,4.2,3.0,11.5",
    "jamaica": "-78.5,17.5,-76.5,19.0",
    "japan": "122.9,24.4,145.8,45.5",
    "jordan": "34.9,29.1,39.3,33.4",
    "kazakhstan": "46.0,40.0,87.0,55.0",
    "kenya": "33.9,-4.6,41.3,5.0",
    "kiribati": "169.2,-3.0,173.7,-2.0",
    "korea, north": "124.3,38.1,130.8,41.4",
    "korea, south": "126.0,33.0,130.8,38.6",
    "kuwait": "46.0,28.6,48.0,30.1",
    "kyrgyzstan": "69.2,39.2,80.0,42.7",
    "laos": "100.0,14.0,107.6,22.0",
    "latvia": "21.0,56.5,28.2,58.8",
    "lebanon": "35.1,33.0,36.6,34.6",
    "lesotho": "27.1,-30.5,29.5,-28.5",
    "liberia": "-9.5,4.3,-7.5,9.5",
    "libya": "9.0,18.0,25.0,33.0",
    "liechtenstein": "9.5,47.1,9.6,47.2",
    "lithuania": "21.0,54.8,26.5,56.5",
    "luxembourg": "5.9,49.5,6.5,50.0",
    "madagascar": "43.5,-25.0,50.5,-12.0",
    "malawi": "33.6,-14.0,35.9,-9.4",
    "malaysia": "99.6,0.9,119.3,7.4",
    "maldives": "72.5,3.1,74.5,7.3",
    "mali": "-12.6,10.2,4.5,23.6",
    "malta": "14.5,35.8,14.6,35.9",
    "marshall islands": "166.0,4.5,173.0,7.5",
    "mauritania": "-17.0,14.5,-10.0,27.5",
    "mauritius": "57.3,-20.5,63.5,-18.0",
    "mexico": "-118.4,14.5,-86.7,32.7",
    "micronesia": "137.0,2.5,165.0,11.0",
    "moldova": "27.2,45.5,30.0,48.5",
    "monaco": "7.4,43.7,7.5,43.8",
    "mongolia": "87.8,41.6,119.9,52.0",
    "montenegro": "18.5,41.9,20.4,42.5",
    "morocco": "-13.0,27.0,-0.5,35.0",
    "mozambique": "30.0,-26.0,40.5,-10.0",
    "myanmar": "92.0,9.5,101.0,28.5",
    "namibia": "11.5,-28.0,25.0,-17.0",
    "nauru": "166.9,-0.5,167.5,-0.1",
    "nepal": "80.0,26.3,88.2,30.4",
    "netherlands": "3.3,50.8,7.0,53.5",
    "new zealand": "166.5,-47.7,178.6,-34.0",
    "nicaragua": "-87.6,11.0,-82.0,15.0",
    "niger": "0.0,11.5,16.0,23.5",
    "nigeria": "2.6,4.2,14.7,13.9",
    "north macedonia": "20.5,41.8,23.0,42.5",
    "norway": "5.0,58.0,31.0,71.5",
    "oman": "56.0,16.5,60.0,26.0",
    "pakistan": "60.9,23.7,77.0,37.1",
    "palau": "131.0,5.0,134.5,8.0",
    "panama": "-77.1,7.5,-77.0,9.0",
    "papua new guinea": "141.0,-11.5,156.0,-3.0",
    "paraguay": "-60.0,-27.0,-54.0,-19.0",
    "peru": "-81.5,-18.5,-68.5,-0.5",
    "philippines": "116.8,4.6,126.6,21.1",
    "poland": "14.1,49.0,24.1,54.9",
    "portugal": "-9.6,36.9,-6.1,42.1",
    "qatar": "50.5,24.5,51.5,26.0",
    "romania": "20.9,43.6,29.7,48.3",
    "russia": "19.6,41.2,180.0,81.9",
    "rwanda": "28.9,-2.1,30.9,2.5",
    "saint kitts and nevis": "-62.8,17.2,-62.6,17.4",
    "saint lucia": "-61.0,13.8,-60.9,14.1",
    "saint vincent and grenadines": "-61.4,12.6,-61.2,13.3",
    "samoa": "-173.0,-14.0,-171.5,-13.0",
    "san marino": "12.5,43.9,12.6,44.0",
    "sao tome and principe": "6.3,0.0,7.0,1.0",
    "saudi arabia": "34.4,16.3,55.7,32.2",
    "senegal": "-17.5,12.0,-11.5,16.0",
    "serbia": "19.0,43.0,23.0,46.5",
    "seychelles": "56.3,-9.2,56.8,-4.5",
    "sierra leone": "-13.5,6.0,-11.0,9.5",
    "singapore": "103.6,1.1,104.0,1.5",
    "slovakia": "16.5,47.7,22.0,49.6",
    "slovenia": "13.3,45.5,16.5,47.5",
    "solomon islands": "155.0,-12.5,160.0,-9.0",
    "somalia": "41.0,-1.5,51.5,12.0",
    "south africa": "16.5,-34.8,32.9,-22.1",
    "south korea": "126.0,33.0,130.8,38.6",
    "south sudan": "24.0,3.5,37.0,13.5",
    "spain": "-9.3,35.9,3.3,43.8",
    "sri lanka": "79.9,5.8,81.5,9.9",
    "sudan": "22.0,8.0,38.5,22.0",
    "suriname": "-58.5,3.9,-53.0,6.0",
    "sweden": "11.0,55.0,24.0,69.0",
    "switzerland": "5.9,45.8,10.5,47.8",
    "syria": "35.5,32.0,42.0,37.5",
    "taiwan": "119.5,20.7,124.5,25.5",
    "tajikistan": "67.0,36.0,79.0,41.0",
    "tanzania": "29.5,-11.5,40.0,-1.5",
    "thailand": "97.3,5.6,105.6,20.4",
    "togo": "0.9,6.0,1.7,11.5",
    "tonga": "-175.0,-22.0,-173.0,-20.0",
    "trinidad and tobago": "-61.8,10.0,-60.6,11.0",
    "tunisia": "7.5,30.0,11.0,37.5",
    "turkey": "26.0,36.0,44.6,42.0",
    "turkmenistan": "52.5,35.0,66.5,42.0",
    "tuvalu": "176.0,-7.0,179.0,-5.0",
    "uganda": "29.5,-1.5,35.0,4.5",
    "ukraine": "22.1,44.4,40.0,52.4",
    "united arab emirates": "51.0,22.5,56.0,26.0",
    "united kingdom": "-8.7,49.9,1.7,60.8",
    "united states": "-125.0,24.5,-66.9,49.3",
    "uruguay": "-58.5,-34.9,-53.2,-30.1",
    "uzbekistan": "56.0,37.0,73.0,45.0",
    "vanuatu": "166.0,-17.7,170.0,-13.0",
    "vatican city": "12.4,41.9,12.5,42.0",
    "venezuela": "-73.0,0.7,-59.0,12.2",
    "vietnam": "102.1,8.1,109.5,23.4",
    "yemen": "42.5,12.0,54.0,19.0",
    "zambia": "22.0,-18.0,33.5,-8.0",
    "zimbabwe": "25.0,-22.0,33.0,-15.0"
}




# Configuration
st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .risk-high { color: #ff0000; font-weight: bold; }
    .risk-medium { color: #ff9900; font-weight: bold; }
    .risk-low { color: #33cc33; font-weight: bold; }
    .map-container { border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
</style>
""", unsafe_allow_html=True)

# Constants
NASA_API_KEY = st.secrets.get("NASA_API_KEY", "8e38a0d0a687d17e4fa97fe161bd567b")
NASA_USERNAME = st.secrets.get("NASA_USERNAME", "sumn0902")
MODEL_FILES = {
    'models': 'fire_risk_models.pkl',
    'scaler': 'fire_risk_scaler.pkl'
}

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        models = joblib.load(MODEL_FILES['models'])
        scaler = joblib.load(MODEL_FILES['scaler'])
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

models, scaler = load_models()

def fetch_fire_data(area="world", days=1):
    """Fetch fire data from NASA FIRMS API"""
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    source = "VIIRS_NOAA20_NRT"
    
    # Validate inputs
    if days < 1 or days > 10:
        st.error("Days must be between 1 and 10")
        return None
    
    # Format area parameter
    if area.lower() == "california":
        area = "-124.5,32.5,-114.0,42.0"  # Bounding box for California
    elif area.lower() != "world":
        # Validate bounding box format
        try:
            west, south, east, north = map(float, area.split(","))
            if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
                st.error("Invalid bounding box coordinates")
                return None
            if west >= east or south >= north:
                st.error("West must be less than east, and south must be less than north")
                return None
        except ValueError:
            st.error("Bounding box must be in format: west,south,east,north (e.g., -124.5,32.5,-114.0,42.0)")
            return None
    
    try:
        url = f"{base_url}/{NASA_API_KEY}/{source}/{area}/{days}"
        st.write("API URL:", url)  # Log URL for debugging
        response = requests.get(url, auth=(NASA_USERNAME, NASA_API_KEY), timeout=30)
        
        if response.status_code == 200:
            # Check if response is a valid CSV
            if not response.text.strip() or "Invalid" in response.text:
                st.error(f"API returned an error: {response.text}")
                return None
            
            data = pd.read_csv(StringIO(response.text))
            st.write("Raw API columns:", data.columns.tolist())  # Log columns
            
            if data.empty:
                st.warning("No fire detections found for the specified area and time")
                return None
            
            # Standardize column names
            column_map = {
                'lat': 'latitude',
                'LATITUDE': 'latitude',
                'lon': 'longitude',
                'LONGITUDE': 'longitude',
                'brightness': 'bright_ti4',
                'bright_ti4_n': 'bright_ti4',
                'bright_ti4_d': 'bright_ti4',
                'fire_radiative_power': 'frp',
                'radiative_power': 'frp',
                'FRP': 'frp'
            }
            data.rename(columns=column_map, inplace=True)
            st.write("Columns after renaming:", data.columns.tolist())  # Log renamed columns
            
            # Verify required columns
            required_cols = ['latitude', 'longitude', 'bright_ti4', 'frp']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return None
                
            return data
        else:
            st.error(f"API Error {response.status_code}: {response.text[:200]}...")
            return None
    except Exception as e:
        st.error(f"Network error: {str(e)}")
        return None

def preprocess_data(data):
    if data is None or data.empty:
        st.error("Input data is None or empty")
        return None
    
    possible_columns = {
        'latitude': ['latitude', 'lat', 'LATITUDE'],
        'longitude': ['longitude', 'lon', 'LONGITUDE'],
        'bright_ti4': ['bright_ti4', 'brightness', 'bright_ti4_n', 'bright_ti4_d'],
        'bright_ti5': ['bright_ti5'],
        'frp': ['frp', 'fire_radiative_power', 'radiative_power', 'FRP'],
        'confidence': ['confidence', 'conf'],
        'acq_date': ['acq_date', 'acquisition_date', 'date']
    }
    
    # Standardize column names
    for standard_name, variants in possible_columns.items():
        for variant in variants:
            if variant in data.columns:
                data.rename(columns={variant: standard_name}, inplace=True)
                break
    
    # Check for required columns
    required_cols = ['latitude', 'longitude', 'bright_ti4', 'frp']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Available columns:", list(data.columns))
        return None
    
    # Impute missing bright_ti5 if needed
    if 'bright_ti5' not in data.columns:
        data['bright_ti5'] = data['bright_ti4'] - 20  # Reasonable default
    
    # Convert numeric columns
    numeric_cols = ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'frp', 'confidence']
    for col in numeric_cols:
        if col in data.columns:
            # Debug: Print sample values before conversion
            if col == 'confidence':
                st.write(f"Sample raw {col} values:", data[col].head().tolist())
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Handle day_of_year
    if 'acq_date' in data.columns:
        # Attempt to parse dates, fallback to current day if parsing fails
        data['acq_date'] = pd.to_datetime(data['acq_date'], errors='coerce')
        invalid_dates = data['acq_date'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"Found {invalid_dates} invalid or unparseable dates in 'acq_date'. Using current day for these rows.")
            data['day_of_year'] = data['acq_date'].dt.dayofyear
            # Fill NaN in day_of_year with current day
            current_day = datetime.now().timetuple().tm_yday
            data['day_of_year'] = data['day_of_year'].fillna(current_day)
        else:
            data['day_of_year'] = data['acq_date'].dt.dayofyear
    else:
        st.warning("'acq_date' column missing. Using current day for day_of_year.")
        data['day_of_year'] = datetime.now().timetuple().tm_yday
    
    # Impute missing or NaN confidence values
    if 'confidence' in data.columns:
        nan_count = data['confidence'].isna().sum()
        if nan_count > 0:
            st.warning(f"Found {nan_count} NaN values in 'confidence'. Imputing with default value 75.")
            data['confidence'] = data['confidence'].fillna(75)
    else:
        st.warning("'confidence' column missing. Using default value 75.")
        data['confidence'] = 75
    
    # Select final features
    features = ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'frp', 'day_of_year', 'confidence']
    
    # Debug: Check for NaN values before dropping
    nan_counts = data[features].isna().sum()
    if nan_counts.any():
        st.write("NaN counts in features before dropna:", nan_counts.to_dict())
    
    # Drop rows with NaN in required features
    data = data[features].dropna()
    if data.empty:
        st.error("No valid data after preprocessing. All rows were dropped due to NaN values.")
        return None
    
    return data

def predict_risk(data, model_name='xgboost'):
    """Make predictions using selected model"""
    if not validate_input(data):
        return None, None
    
    try:
        model = models[model_name]
        
        # Ensure 2D array even for single prediction
        if len(data.shape) == 1:
            data = data.values.reshape(1, -1) if isinstance(data, pd.Series) else np.array(data).reshape(1, -1)
        
        # Scale if logistic regression
        if model_name == 'logistic_regression':
            data = scaler.transform(data)
            
        predictions = model.predict(data)
        
        # Handle probability predictions
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)
            # Ensure probabilities are for all classes (some models might drop classes)
            if proba.shape[1] < len(model.classes_):
                full_proba = np.zeros((len(predictions), len(model.classes_)))
                for i, cls in enumerate(model.classes_):
                    if cls in model.classes_:
                        full_proba[:, i] = proba[:, np.where(model.classes_ == cls)[0][0]]
                proba = full_proba
        
        return predictions, proba
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Problematic data shape:", data.shape if hasattr(data, 'shape') else type(data))
        return None, None
    
def validate_input(data):
    """Ensure data is suitable for prediction"""
    if data is None or data.empty:
        return False
    
    # Check array dimensions
    if hasattr(data, 'shape'):
        if len(data.shape) == 1 and data.shape[0] <= 1:
            st.error("Input data needs multiple features for prediction")
            return False
    
    # Rest of validation checks...
    return True
def plot_interactive_map(data, predictions):
    """Create interactive Plotly map visualization"""
    if data is None or predictions is None:
        return
        
    risk_labels = ["Low", "Moderate", "High", "Very High", "Extreme"]
    data['risk'] = predictions
    data['risk_label'] = data['risk'].map(dict(enumerate(risk_labels)))
    
    fig = px.scatter_mapbox(
        data,
        lat="latitude",
        lon="longitude",
        color="risk_label",
        color_discrete_sequence=["green", "yellow", "orange", "red", "darkred"],
        zoom=3,
        height=600,
        hover_data=["bright_ti4", "frp"]
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig, use_container_width=True, className="map-container")

# Sidebar
with st.sidebar:
    st.title("Navigation")
    app_mode = st.radio(
        "Select Mode",
        ["Real-time Monitoring", "Historical Analysis", "Single Location Prediction"],
        index=0
    )
    
    st.title("Model Settings")
    model_name = st.selectbox(
        "Prediction Model",
        ["xgboost", "random_forest", "logistic_regression"],
        index=0
    )

# Main App
st.title("🔥 Forest Fire Prediction System")
st.markdown("Predict Forest Fire using NASA satellite data and machine learning")

if app_mode == "Real-time Monitoring":
    st.subheader("Live Fire Risk Monitoring")
    
    country = st.text_input("Enter Country Name (e.g., India, USA, Brazil)").lower()
    days = st.number_input("Days of data", 1, 7, 1)
    
    if country in country_bounding_boxes:
        bbox = country_bounding_boxes[country]
        st.success(f"Using bounding box for {country.title()}: {bbox}")
        
        if st.button("Analyze Current Data"):
            with st.spinner("Fetching NASA data..."):
                data = fetch_fire_data(bbox, days)
                if data is not None:
                    processed = preprocess_data(data)
                    predictions, _ = predict_risk(processed, model_name)

                    st.success(f"Analyzed {len(data)} fire detections")
                    tab1, tab2 = st.tabs(["Map", "Data"])

                    with tab1:
                        plot_interactive_map(processed, predictions)

                    with tab2:
                        st.dataframe(processed)
    else:
        st.warning("Country not found in database. Please try India, USA, Brazil, etc.")

    

elif app_mode == "Historical Analysis":
    st.header("Historical Risk Patterns")
    
    # Input for coordinates (bounding box) and dates
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime(2025, 3, 1))  # Default start date
    with col2:
        end_date = st.date_input("End date", datetime(2025, 3, 5))  # Default end date
    
    # Input for region (latitude and longitude or bounding box)
    st.subheader("Enter Coordinates or Bounding Box")
    lat = st.number_input("Latitude (°)", -90.0, 90.0, 34.05)  # Default to California (Los Angeles)
    lon = st.number_input("Longitude (°)", -180.0, 180.0, -118.25)  # Default to California (Los Angeles)
    
    # Option for custom bounding box (for flexibility)
    use_bounding_box = st.checkbox("Use custom bounding box (latitude_min, longitude_min, latitude_max, longitude_max)")
    bounding_box = None
    if use_bounding_box:
        min_lat = st.number_input("Min Latitude", -90.0, 90.0, 32.5)  # Default to California bounding box
        min_lon = st.number_input("Min Longitude", -180.0, 180.0, -124.5)
        max_lat = st.number_input("Max Latitude", -90.0, 90.0, 42.0)
        max_lon = st.number_input("Max Longitude", -180.0, 180.0, -114.0)
        bounding_box = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    if st.button("Analyze History"):
        days = (end_date - start_date).days
        if days < 1 or days > 10:
            st.error("Date range must be between 1 and 10 days.")
        elif start_date > datetime.now().date():
            st.error("Future dates are not supported by the API.")
        else:
            # Choose the correct area parameter (coordinates or bounding box)
            area = bounding_box if use_bounding_box else f"{lon},{lat},{lon},{lat}"
            
            with st.spinner(f"Processing {days} days of data..."):
                data = fetch_fire_data(area, days)
                if data is not None:
                    processed = preprocess_data(data)
                    if processed is not None:
                        predictions, _ = predict_risk(processed, model_name)
                        if predictions is not None:
                            st.plotly_chart(px.line(
                                processed.assign(
                                    date=pd.to_datetime(data['acq_date'], errors='coerce'),
                                    risk=predictions
                                ).groupby('date')['risk'].mean().reset_index(),
                                x='date', y='risk',
                                title="Average Daily Fire Risk"
                            ))
                        else:
                            st.error("Prediction failed. Check model compatibility or input data.")
                    else:
                        st.error("Data preprocessing failed. Check API response and column names.")
                else:
                    st.error("Failed to fetch data. Check API credentials, area, or parameters.")


elif app_mode == "Single Location Prediction":
    st.header("Location-Specific Assessment")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", -90.0, 90.0, 34.05)
            lon = st.number_input("Longitude", -180.0, 180.0, -118.25)
        with col2:
            temp = st.number_input("Temperature (K)", 200.0, 500.0, 320.0)
            frp = st.number_input("FRP (MW)", 0.0, 1000.0, 10.0)
        
        if st.form_submit_button("Predict Risk"):
            input_data = pd.DataFrame([{
                'latitude': lat,
                'longitude': lon,
                'bright_ti4': temp,
                'bright_ti5': temp - 20,
                'frp': frp,
                'day_of_year': datetime.now().timetuple().tm_yday,
                'confidence': 75
            }])
            
            prediction, proba = predict_risk(input_data, model_name)
            if prediction is not None:
                risk_level = ["Low", "Moderate", "High", "Very High", "Extreme"][prediction[0]]
                st.markdown(f"### Predicted Risk: <span class='risk-{risk_level.lower().split()[0]}'>{risk_level}</span>", 
                           unsafe_allow_html=True)
                
                if proba is not None:
                    st.plotly_chart(px.bar(
                        x=["Low", "Moderate", "High", "Very High", "Extreme"],
                        y=proba[0],
                        title="Risk Probability Distribution"
                    ))