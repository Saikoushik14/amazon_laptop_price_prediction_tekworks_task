import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# ------------------------------
# 1. Load & Clean Dataset
# ------------------------------
@st.cache_data # Cache data so it doesn't reload on every click
def load_data():
    df = pd.read_csv("amazon_laptop_price_dataset.csv")

    # Convert USD to INR
    df['Price_USD'] = pd.to_numeric(df['Price_USD'], errors='coerce')
    df['Price_INR'] = df['Price_USD'] * 90.94

    # Remove outliers (IQR method)
    Q1 = df['Price_INR'].quantile(0.25)
    Q3 = df['Price_INR'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Price_INR'] >= (Q1 - 1.5 * IQR)) & (df['Price_INR'] <= (Q3 + 1.5 * IQR))]

    # Filter OS
    df = df[df['Operating_System'].isin(['macOS', 'Windows 10', 'Windows 11'])]

    # Map categorical features to numbers
    processor_map = {'Intel i3': 1, 'Intel i5': 2, 'Intel i7': 3, 'AMD Ryzen 3': 1, 'AMD Ryzen 5': 2, 'AMD Ryzen 7': 3}
    gpu_map = {'Integrated': 0, 'AMD Radeon': 1, 'NVIDIA GTX 1650': 2, 'NVIDIA RTX 3050': 3}
    os_map = {'Windows 10': 1, 'Windows 11': 2, 'macOS': 3}

    df['Processor'] = df['Processor'].map(processor_map)
    df['GPU'] = df['GPU'].map(gpu_map)
    df['Operating_System'] = df['Operating_System'].map(os_map)
    
    return df

df_raw = load_data()

# ------------------------------
# 2. Preprocessing & Training
# ------------------------------
# OneHotEncode Brand
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
brand_encoded = encoder.fit_transform(df_raw[['Brand']])
brand_cols = encoder.get_feature_names_out(['Brand'])
brand_df = pd.DataFrame(brand_encoded, columns=brand_cols, index=df_raw.index)

# Define final feature set
base_features = ['Processor', 'RAM_GB', 'Storage_GB', 'Operating_System', 'GPU', 'Rating']
X = pd.concat([df_raw[base_features], brand_df], axis=1)
y = df_raw['Price_INR']

# Train Model
model = LinearRegression()
model.fit(X, y)

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.set_page_config(page_title="Laptop Predictor", page_icon="💻")
st.title("💻 Laptop Price Prediction App")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Select Brand", encoder.categories_[0])
    processor = st.selectbox("Select Processor", ['Intel i3','Intel i5','Intel i7','AMD Ryzen 3','AMD Ryzen 5','AMD Ryzen 7'])
    ram = st.selectbox("Select RAM (GB)", sorted(df_raw['RAM_GB'].unique()))

with col2:
    storage = st.selectbox("Select Storage (GB)", sorted(df_raw['Storage_GB'].unique()))
    os_choice = st.selectbox("Select Operating System", ['Windows 10','Windows 11','macOS'])
    gpu_choice = st.selectbox("Select GPU", ['Integrated','AMD Radeon','NVIDIA GTX 1650','NVIDIA RTX 3050'])

rating = st.slider("User Rating", 1.0, 5.0, 4.2)

# ------------------------------
# 4. Prediction Logic
# ------------------------------
# Map UI inputs to match training data
processor_map = {'Intel i3':1,'Intel i5':2,'Intel i7':3,'AMD Ryzen 3':1,'AMD Ryzen 5':2,'AMD Ryzen 7':3}
gpu_map = {'Integrated':0,'AMD Radeon':1,'NVIDIA GTX 1650':2,'NVIDIA RTX 3050':3}
os_map = {'Windows 10':1,'Windows 11':2,'macOS':3}

# Build the input dictionary
input_dict = {
    'Processor': processor_map[processor],
    'RAM_GB': ram,
    'Storage_GB': storage,
    'Operating_System': os_map[os_choice],
    'GPU': gpu_map[gpu_choice],
    'Rating': rating
}

# Add the brand one-hot columns (1 for the selected brand, 0 for others)
for col in brand_cols:
    input_dict[col] = 1.0 if col == f"Brand_{brand}" else 0.0

# Convert to DataFrame and REORDER columns to match X exactly
input_df = pd.DataFrame([input_dict])
input_df = input_df[X.columns] 

st.markdown("---")
if st.button("Calculate Price", use_container_width=True):
    prediction = model.predict(input_df)[0]
    
    # Simple logic to ensure we don't show negative prices
    final_price = max(0, prediction)
    
    st.balloons()
    st.success(f"### Estimated Price: ₹ {final_price:,.2f}")
    
    # Show technical details in an expander
    with st.expander("See Feature breakdown"):
        st.write("The model used the following encoded features:")
        st.dataframe(input_df)