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
with open('X1.pkl','rb') as f:
    X = pickle.load(f)

st.title("Singapore Resale Flat Price Predicting")

# Load your data
with open('all_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

with open('le1.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler= pickle.load(f)

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
    all_data['remaining_lease'] = pd.to_numeric(all_data['remaining_lease'],errors='coerce') 
    
    all_data['flat_model'] = all_data['flat_model'].replace({
        'New Generation': 'NEW GENERATION',
        'Improved': 'IMPROVED',
        'Model A': 'MODEL A',
        'Standard': 'STANDARD',
        'Apartment': 'APARTMENT',
        'Simplified': 'SIMPLIFIED',
        'Model A-Maisonette': 'MODEL A-MAISONETTE',
        'Maisonette': 'MAISONETTE',
        'Multi Generation': 'MULTI GENERATION',
        'Improved-Maisonette': 'IMPROVED-MAISONETTE',
        'Premium Apartment': 'PREMIUM APARTMENT',
        '2-room': '2-ROOM'
    })
    
    #Features related to months
    all_data['month'] = pd.to_datetime(all_data['month'], format = '%Y-%m')
    all_data['year'] = all_data['month'].dt.year
    all_data['month_number'] = all_data['month'].dt.month
    all_data['month_sin'] = np.sin(2 * np.pi * all_data['month_number'] / 12)
    all_data['month_cos'] = np.cos(2 * np.pi * all_data['month_number'] / 12)
    # Calculate the elapsed time in months as an integer
    all_data['elapsed_time'] = (all_data['month'].dt.year - all_data['month'].dt.year.min()) * 12 + (all_data['month'].dt.month - all_data['month'].dt.month.min())
    
    return all_data



# Create Streamlit form
with st.form(key='predict_form'):
    # Define three columns

    input_data = {}
    first_row = loaded_data.iloc[1000]
    cols = st.columns(3)  # Create 3 columns

    for i, col in enumerate(loaded_data.columns):
        col_index = i % 3  # Determine column index (0, 1, or 2)
        with cols[col_index]:
            if col in ['town', 'flat_type', 'block', 'storey_range', 'flat_model','street_name']:  # Categorical fields
                # Select box for categorical fields
                #options = label_encoders[col].classes_.tolist()
                options = loaded_data[col].unique().tolist()
                # Handle NaN values
                default_value = first_row[col] if pd.notna(first_row[col]) else options['']
                input_data[col] = st.selectbox(col, options, index=options.index(default_value))
            else:  # Special handling for 'country'
                if col == 'month':
                    input_data[col] = st.text_input(col, value=str(first_row[col]))
                elif col == 'resale_price':
                    continue
                elif col == 'lease_commence_date':
                    input_data[col] = st.text_input(col, value=int(first_row[col]))
                elif col == 'remaining_lease':
                    input_data[col] = st.text_input(col, value=str(first_row[col]))
                else:
                    input_data[col] = st.text_input(col, value=float(first_row[col]))

    # Submit button
    submit_button = st.form_submit_button(label='Predict Resale Price')

input_df = pd.DataFrame([input_data])

# Handle empty inputs by replacing with default values
input_df.replace('', np.nan, inplace=True)
input_df.fillna(0, inplace=True)
#st.dataframe(input_df)
input_df = features_engg(input_df)
#st.dataframe(input_df)

categorical_features = ['town', 'flat_type', 'block', 'storey_range', 'flat_model','street_name']

# Apply Label Encoding to each categorical feature
for column in categorical_features:
    if column in input_df.columns:
        le = label_encoders[column]
        input_df[column] = input_df[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)


if submit_button:
    
    prediction = model.predict(input_df[X.columns])
    #prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    #st.write(prediction)
    st.markdown(f'<p style="color:green; font-weight:bold; font-size:24px; background-color:lightyellow; padding:10px;">Predicted Selling Price: {prediction[0]:.2f}</p>', unsafe_allow_html=True)

def image_to_base64(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path to your local image
image_path = "pngegg (1).png"
  # Update this path to your local image file
base64_image = image_to_base64(image_path)

# Set background image using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
