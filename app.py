import streamlit as st
import pandas as pd
import pickle

# Load the model using pickle
with open('best_gb_regressor.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize session state for storing predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Streamlit app title and configuration
st.set_page_config(page_title='Real Estate Price Prediction System', layout='wide')

# Custom CSS for dark mode and styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    
    /* Title container styling */
    .title-container {
        background-color: #90EE90;
        color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Predict button container */
    .predict-container {
        background-color: #90EE90;
        color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Column headers */
    .column-header {
        color: #90EE90;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* Input containers */
    .input-section {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #404040;
    }
    
    /* Override Streamlit's default styling */
    .stSelectbox label, .stSlider label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2d2d2d;
        color: white;
        border: 2px solid #90EE90;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #90EE90;
        color: #1e1e1e;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        color: white;
    }
    
    /* Sidebar headers */
    .sidebar-header {
        color: #90EE90;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
        padding: 10px;
        background-color: #1e1e1e;
        border-radius: 5px;
    }
    
    /* Investment option containers */
    .investment-container {
        background-color: #1a1a1a;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #90EE90;
        border: 1px solid #404040;
    }
    
    /* Property feature list styling */
    .feature-list {
        background-color: #1a1a1a;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-family: monospace;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid #404040;
        white-space: pre-line;
    }
    
    /* Clear predictions button */
    .clear-button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background-color: #90EE90;
        color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown("""
<div class="title-container">
    <h1> Real Estate Price Prediction System</h1>
    <p>Get accurate property price predictions using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for investment comparison
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 Compare Investment Opportunities</div>', unsafe_allow_html=True)
    
    if len(st.session_state.predictions) == 0:
        st.info("Make some predictions first to compare investment opportunities!")
    else:
        st.write(f"**Total Predictions Made:** {len(st.session_state.predictions)}")
        
        # Investment preference selection
        st.markdown("### Select Investment Strategy")
        
        # Custom label with black text
        st.markdown('<p style="color: black; font-weight: bold; margin-bottom: 5px;">Choose your investment preference:</p>', unsafe_allow_html=True)
        
        investment_choice = st.selectbox(
            "Choose your investment preference:",
            ["Select an option", "Minimal Capital Investment", "High Capital Investment"],
            key="investment_selectbox",
            label_visibility="collapsed"
        )
        
        if investment_choice != "Select an option":
            # Find lowest and highest priced properties
            prices = [pred['price'] for pred in st.session_state.predictions]
            
            if investment_choice == "Minimal Capital Investment":
                min_price_idx = prices.index(min(prices))
                selected_property = st.session_state.predictions[min_price_idx]
                investment_type = "💰 Minimal Capital Investment"
                investment_description = "Lowest priced property from your predictions"
            else:  # High Capital Investment
                max_price_idx = prices.index(max(prices))
                selected_property = st.session_state.predictions[max_price_idx]
                investment_type = "🏆 High Capital Investment"
                investment_description = "Highest priced property from your predictions"
            
            # Display selected investment option
            st.markdown(f"""
            <div class="investment-container">
                <h4>{investment_type}</h4>
                <p style="color: #cccccc; margin-bottom: 10px;">{investment_description}</p>
                <h3 style="color: #90EE90;">${selected_property['price']:,.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display property features (removed waterfront)
            st.markdown("**Property Features:**")
            features_text = f"""bathrooms: {selected_property['features']['bathrooms']}
bedrooms: {selected_property['features']['bedrooms']}
sqft_living: {selected_property['features']['sqft_living']:,}
sqft_lot: {selected_property['features']['sqft_lot']:,}
floors: {selected_property['features']['floors']}
view: {selected_property['features']['view']}
condition: {selected_property['features']['condition']}
sqft_above: {selected_property['features']['sqft_above']:,}
sqft_basement: {selected_property['features']['sqft_basement']:,}
yr_built: {selected_property['features']['yr_built']}
yr_renovated: {selected_property['features']['yr_renovated']}
city: {selected_property['features']['city']}
price_per_sqft: ${selected_property['features']['price_per_sqft']:.2f}
lot_to_living_ratio: {selected_property['features']['lot_to_living_ratio']:.2f}"""
            
            st.markdown(f'<div class="feature-list">{features_text}</div>', unsafe_allow_html=True)
        
        # Clear predictions button
        st.markdown("---")
        if st.button("🗑️ Clear All Predictions", key="clear_predictions"):
            st.session_state.predictions = []
            st.rerun()

# City mapping
city_mapping = {
    "Shoreline": 1, "Kent": 2, "Bellevue": 3, "Redmond": 4, "Seattle": 5,
    "Maple Valley": 6, "North Bend": 7, "Lake Forest Park": 8, "Sammamish": 9,
    "Auburn": 10, "Des Moines": 11, "Bothell": 12, "Federal Way": 13,
    "Kirkland": 14, "Issaquah": 15, "Woodinville": 16, "Normandy Park": 17,
    "Fall City": 18, "Renton": 19, "Carnation": 20, "Snoqualmie": 21,
    "Duvall": 22, "Burien": 23, "Covington": 24, "Inglewood-Finn Hill": 25,
    "Kenmore": 26, "Newcastle": 27, "Black Diamond": 28, "Ravensdale": 29,
    "Clyde Hill": 30, "Algona": 31, "Mercer Island": 32, "Skykomish": 33,
    "Tukwila": 34, "Vashon": 35, "SeaTac": 36, "Enumclaw": 37,
    "Snoqualmie Pass": 38, "Pacific": 39, "Beaux Arts Village": 40,
    "Preston": 41, "Milton": 42, "Yarrow Point": 43, "Medina": 44
}

# Create two columns for the layout
col1, col2 = st.columns(2)

# Left column - Sliders
with col1:
    st.markdown('<div class="column-header"> Property Measurements</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        sqft_living = st.slider('Square Feet of Living Space', 0, 10000, 2000)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        sqft_lot = st.slider('Square Feet of Lot', 0, 100000, 10000)
        st.markdown('</div>', unsafe_allow_html=True)
    

    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        sqft_above = st.slider('Square Feet Above Ground', 0, 10000, 2000)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        sqft_basement = st.slider('Square Feet of Basement', 0, 5000, 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        yr_built = st.slider('Year Built', 1900, 2020, 2000)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        yr_renovated = st.slider('Year Renovated', 1900, 2020, 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        price_per_sqft_input = st.slider('Price per Square Foot ($)', 50, 2000, 300)
        st.markdown('</div>', unsafe_allow_html=True)

# Right column - Select boxes (removed waterfront)
with col2:
    st.markdown('<div class="column-header">Property Features</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        bedrooms = st.selectbox('Number of Bedrooms', list(range(0, 11)), index=3)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        bathrooms_options = ["0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]
        bathrooms = st.selectbox('Number of Bathrooms', bathrooms_options, index=3)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        view = st.selectbox('View Rating (0-4)', list(range(5)), index=0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        condition = st.selectbox('Property Condition (1-5)', list(range(1, 6)), index=2)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        floors = st.selectbox('Number of Floors', list(range(1, 5)), index=0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        city = st.selectbox('Select City', list(city_mapping.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

# Predict button with custom styling
st.markdown('<div class="predict-container">', unsafe_allow_html=True)
predict_clicked = st.button(' Predict Property Price')
st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic and display
if predict_clicked:
    # Convert bathrooms from string to float
    bathrooms_float = float(bathrooms)
    
    # Extract city number from mapping
    city_number = city_mapping[city]
    
    # PREPROCESSING: Create sqft_living_above by averaging sqft_living and sqft_above
    sqft_living_above = (sqft_living + sqft_above) / 2
    
    # Create a DataFrame with preprocessed input data
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms_float],
        'sqft_living_above': [sqft_living_above],  # New averaged column
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'view': [view],
        'condition': [condition],
        'sqft_basement': [sqft_basement],
        'yr_built': [yr_built],
        'yr_renovated': [yr_renovated],
        'city': [city_number],
        'price_per_sqft': [price_per_sqft_input]  # Use input value
    })
    
    # Calculate lot_to_living_ratio
    input_data['lot_to_living_ratio'] = input_data['sqft_lot'] / input_data['sqft_living_above']
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_price = prediction[0]
    
    # Get the calculated lot_to_living_ratio
    lot_to_living_ratio = input_data['lot_to_living_ratio'].iloc[0]
    
    # Store prediction in session state with calculated values
    prediction_data = {
        'price': predicted_price,
        'price_per_sqft': price_per_sqft_input,
        'lot_to_living_ratio': lot_to_living_ratio,
        'features': {
            'bathrooms': bathrooms_float,
            'bedrooms': bedrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'view': view,
            'condition': condition,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated,
            'city': city,
            'price_per_sqft': price_per_sqft_input,
            'lot_to_living_ratio': lot_to_living_ratio
        }
    }
    
    st.session_state.predictions.append(prediction_data)
    
    predicted_price_formatted = f"${predicted_price:,.2f}"
    
    # Display the prediction with custom styling
    st.markdown(f"""
    <div class="prediction-result">
        <h2>💰 Predicted Property Price</h2>
        <h1>{predicted_price_formatted}</h1>
        <p>Based on the selected property features and location</p>
        <small>Prediction #{len(st.session_state.predictions)} saved for comparison</small>
    </div>
    """, unsafe_allow_html=True)
    