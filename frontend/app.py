# app.py (Streamlit Frontend)
import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🤗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .stTextInput > div > div > input {
            font-size: 18px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Backend URL - replace with your FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

def analyze_sentiment(text):
    """Send text to FastAPI backend for analysis"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze-sentiment",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def main():
    st.title("📝 Sentiment Analysis")
    st.write("Enter your text below to analyze its sentiment.")
    
    # Text input
    text_input = st.text_area(
        "Input Text",
        height=150,
        placeholder="Enter your text here..."
    )
    
    # Add a predict button
    if st.button("Analyze", type="primary"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing text..."):
            result = analyze_sentiment(text_input)
            
            if "error" in result:
                st.error(f"Error during analysis: {result['error']}")
            else:
                # Display results
                col1, col2 = st.columns(2)
                
                # Set color based on sentiment
                sentiment_color = "#198754" if result["label"] == "Positive" else "#dc3545"
                
                with col1:
                    st.markdown("### Sentiment")
                    st.markdown(
                        f"<div class='prediction-box' style='background-color: {sentiment_color}; color: white;'>"
                        f"<h2 style='margin: 0;'>{result['label']}</h2>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown("### Confidence Score")
                    st.markdown(
                        f"<div class='prediction-box' style='background-color: #f0f2f6'>"
                        f"<h2 style='color: #0066cc; margin: 0;'>{result['confidence']:.4f}</h2>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                
                # Add JSON output
                st.markdown("### Raw Output")
                st.code(json.dumps(result, indent=2), language="json")

if __name__ == "__main__":
    main()