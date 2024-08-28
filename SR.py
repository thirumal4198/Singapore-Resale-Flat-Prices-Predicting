import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import base64

# Load the trained model
with open('DT1.pkl', 'rb') as file:
    model = pickle.load(file)

with open('X_columns.pkl', 'rb') as f:
    X = pickle.load(f)

# Custom CSS for shadow effect on title
st.markdown(
    """
    <style>
    .custom-title {
        text-shadow: 2px 2px 5px rgba(255, 255, 255, 2);
        color: rgb(230,34,185);
        font-size: 4em;
        font-weight: bold;
        margin-bottom: 0em;
    }
    .custom-label {
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(255, 255, 255, 0.5);
        color: rgb(255,0,0);
        font-size: 1.2em;
        margin-bottom: 0em;
    }
    .custom-write {
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(255, 255, 255, 0.5);
        color: rgb(73,44,76);
        font-size: 1.2em;
        margin-bottom: 0em;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title with shadow effect
st.markdown('<h1 class="custom-title">Singapore Resale Flat Price Predicting</h1>', unsafe_allow_html=True)
st.write('<h2 class="custom-write">Check the exact rent or sale value for a property within seconds</h2>', unsafe_allow_html=True)
st.write('-----')

# Load your data
with open('loaded_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

with open('le1.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('option_dict.pkl','rb') as f:
    option_dict = pickle.load(f)

def features_engg(all_data):
    def convert_years_to_months(lease_str):
        if isinstance(lease_str, str):
            if lease_str == 'nan':
                return np.nan
            else:
                parts = lease_str.split(' ')
                year = int(parts[0])
                months = 0
                if len(parts) > 2:
                    months = int(parts[2])
                return year*12 + months
        return lease_str

    all_data['remaining_lease'] = all_data['remaining_lease'].fillna(0)
    all_data['remaining_lease'] = all_data['remaining_lease'].apply(convert_years_to_months)
    all_data['remaining_lease'] = pd.to_numeric(all_data['remaining_lease'], errors='coerce') 
    
    
    
    # Features related to months
    all_data['month'] = pd.to_datetime(all_data['month'], format='%Y-%m')
    all_data['year'] = all_data['month'].dt.year
    all_data['month_number'] = all_data['month'].dt.month
    all_data['month_sin'] = np.sin(2 * np.pi * all_data['month_number'] / 12)
    all_data['month_cos'] = np.cos(2 * np.pi * all_data['month_number'] / 12)
    
    # Calculate the elapsed time in months as an integer
    loaded_data['month'] = pd.to_datetime(loaded_data['month'], format='%Y-%m')
    min_month = loaded_data['month'].min()
    all_data['elapsed_time'] = ((all_data['month'] - min_month).dt.days // 30).astype(int)

    return all_data

# Header for prediction
# Custom CSS for shadow effect on header
st.markdown(
    """
    <style>
    .custom-header {
        text-shadow: 2px 2px 5px rgba(255, 255, 255, 2);
        color: rgb(255,0,105);
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 0em;
    }
    </style>
    """, unsafe_allow_html=True
)

# Header with shadow effect
st.markdown('<h2 class="custom-header">Lets calculate the value for you</h2>', unsafe_allow_html=True)


# Create Streamlit form for prediction

# Create Streamlit form for prediction
with st.form(key='predict_form'):
    # Define three columns
    input_data = {}
    first_row = loaded_data.iloc[0]
    
    # Create 2 columns for layout
    cols = st.columns(2)

    # First column inputs
    with cols[0]:
        st.markdown('<div class="custom-label">Month (Ex: YYYY-MM)</div>', unsafe_allow_html=True)
        input_data['month'] = st.text_input(
            '', value=str(first_row['month'])
        )
        st.markdown('<div class="custom-label">Street Name</div>', unsafe_allow_html=True)
        input_data['street_name'] = st.selectbox(
            '', loaded_data['street_name'].unique().tolist(),
            index=loaded_data['street_name'].unique().tolist().index(first_row['street_name'])
        )
        
        st.markdown('<div class="custom-label">Flat Type</div>', unsafe_allow_html=True)
        input_data['flat_type'] = st.selectbox(
            '', loaded_data['flat_type'], 
            index=loaded_data['flat_type'].unique().tolist().index(first_row['flat_type'])
        )
        st.markdown('<div class="custom-label">Floor Area (sqm)</div>', unsafe_allow_html=True)
        input_data['floor_area_sqm'] = st.text_input(
            '', value=float(first_row['floor_area_sqm'])
        )
        
        st.markdown('<div class="custom-label">Lease Commence Date (Ex: YYYY)</div>', unsafe_allow_html=True)
        input_data['lease_commence_date'] = st.text_input(
            '', value=int(first_row['lease_commence_date'])
        )
        
    # Second column inputs
    with cols[1]:
        st.markdown('<div class="custom-label">Town</div>', unsafe_allow_html=True)
        input_data['town'] = st.selectbox(
            '', option_dict['town'], 
            index=loaded_data['town'].unique().tolist().index(first_row['town'])
        )
        st.markdown('<div class="custom-label">Block</div>', unsafe_allow_html=True)
        input_data['block'] = st.selectbox(
            '', option_dict['block'],
            index=loaded_data['block'].unique().tolist().index(first_row['block'])
        )
        st.markdown('<div class="custom-label">Flat Model</div>', unsafe_allow_html=True)
        input_data['flat_model'] = st.selectbox(
            '', option_dict['flat_model'], 
            index=loaded_data['flat_model'].unique().tolist().index(first_row['flat_model'])
        )
        
        st.markdown('<div class="custom-label">Storey Range</div>', unsafe_allow_html=True)
        input_data['storey_range'] = st.selectbox(
            '', option_dict['storey_range'], 
            index=loaded_data['storey_range'].unique().tolist().index(first_row['storey_range'])
        )
       
        st.markdown('<div class="custom-label">Remaining Lease</div>', unsafe_allow_html=True)
        lease_cols = st.columns(2)

        with lease_cols[0]:
            st.markdown('<div class="custom-label">Years (0 - 999)</div>', unsafe_allow_html=True)
            remaining_lease_years = st.number_input('', min_value=0, max_value=999, value=0, step=1, placeholder="Years")
        with lease_cols[1]:
            st.markdown('<div class="custom-label">Months (0 - 11)</div>', unsafe_allow_html=True)
            remaining_lease_months = st.number_input('', min_value=0, max_value=11, value=0, step=1, placeholder="Months")
        
        # Combine years and months into total months
        total_remaining_lease_months = remaining_lease_years * 12 + remaining_lease_months
        input_data['remaining_lease'] = total_remaining_lease_months
    # Submit button
    submit_button = st.form_submit_button(label='Predict Resale Price')

input_df = pd.DataFrame([input_data])

# Handle empty inputs by replacing them with default values
input_df.replace('', np.nan, inplace=True)
input_df.fillna(0, inplace=True)

# Apply feature engineering
input_df = features_engg(input_df)

categorical_features = ['town', 'flat_type', 'block', 'storey_range', 'flat_model', 'street_name']

# Apply Label Encoding to each categorical feature
for column in categorical_features:
    if column in input_df.columns:
        le = label_encoders[column]
        input_df[column] = input_df[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)


if submit_button:
    prediction = model.predict(input_df[X])
    st.write('-----')
    st.markdown(
        f'<p style="color:yellow; font-weight:bold; font-size:50px; background-color:rgb(100, 160, 206); padding:10px;">Predicted Selling Price: S$ {prediction[0]:.2f}</p>',
        unsafe_allow_html=True
    )


def image_to_base64(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path to your local image
image_path = "IMG.png"  # Update this path to your local image file
base64_image = image_to_base64(image_path)

# Apply background image using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: 50em auto;
        background-repeat: no-repeat;
        background-attachment: fixed;
        /* Ensure the background is positioned correctly */
        background-position: center;
        color: #000; /* Default text color to ensure visibility */
    }}
    .streamlit-expanderHeader {{
        color: #000; /* Ensure visibility of expander headers if used */
    }}
    .streamlit-expanderContent {{
        color: #000; /* Ensure visibility of expander content if used */
    }}
    </style>
    """,
    unsafe_allow_html=True
)
