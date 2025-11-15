import streamlit as st
import requests
import time
from pathlib import Path

# API Configuration
API_BASE = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="SoundMatch AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS from external file
def load_css():
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ style.css file not found. Using default styling.")

load_css()

# Initialize Session State
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# API Helper Functions
def check_api_health():
    """Check if API is available and healthy"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status", "unknown"), data.get("models_loaded", False)
        return "offline", False
    except:
        return "offline", False

def get_recommendations_from_audio(file_bytes, filename, top_n=10):
    """Upload audio file and get recommendations"""
    try:
        files = {"file": (filename, file_bytes, "audio/mpeg")}
        params = {"top_n": top_n}
        response = requests.post(
            f"{API_BASE}/recommend/from-audio",
            files=files,
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>🎵 Navigation</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "",
        ["🏠 Home", "🎧 Upload & Recommend", "⭐ Favorites", "ℹ️ About"],
        key="navigation"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")
    status, models_loaded = check_api_health()
    
    if status == "healthy":
        st.success("✅ API Online")
    elif status == "degraded":
        st.warning("⚠️ API Degraded")
    else:
        st.error("❌ API Offline")
    
    st.metric("Favorites", len(st.session_state["favorites"]))

# Page 1: Home / Dashboard
if page == "🏠 Home":
    st.markdown("<h1 class='main-title gradient-text'>🎵 SoundMatch AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Music Discovery Through Audio Analysis</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='card fade-in' style='text-align: center;'>
            <h2 style='color: #b39ddb;'>🎶 Welcome to the Future of Music Discovery</h2>
            <p style='font-size: 1.1rem; color: #7b8794; margin: 20px 0;'>
                Upload any audio clip and let our AI find your perfect musical matches using 
                advanced MFCC feature extraction and hybrid machine learning models.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Status Section
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>🔌 System Status</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    if st.button("🔄 Check API Status", use_container_width=True):
        with st.spinner("Checking..."):
            status, models_loaded = check_api_health()
    else:
        status, models_loaded = check_api_health()
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if status == "healthy":
            st.markdown("<div class='status-badge status-healthy'>✅ Healthy</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 15px;'>All systems operational</p>", unsafe_allow_html=True)
        elif status == "degraded":
            st.markdown("<div class='status-badge status-degraded'>⚠️ Degraded</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 15px;'>Some features limited</p>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-badge status-offline'>❌ Offline</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 15px;'>Cannot reach API</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if models_loaded:
            st.markdown("<div class='status-badge status-healthy'>🤖 Models Loaded</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 15px;'>AI ready to recommend</p>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-badge status-offline'>🤖 Models Missing</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 15px;'>Check model files</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='status-badge' style='background: linear-gradient(90deg, #d4a5d4 0%, #f5a8c8 100%); color: white;'>💾 Database</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; margin-top: 15px; color: #7b8794;'>Spotify tracks ready</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Grid
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='card glow-on-hover' style='text-align: center;'>
            <h1 style='color: #d4a5d4;'>🎵</h1>
            <h4 style='color: #5a4a6a;'>Audio Upload</h4>
            <p style='font-size: 0.9rem; color: #7b8794;'>Support for MP3, WAV, M4A formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card glow-on-hover' style='text-align: center;'>
            <h1 style='color: #f5a8c8;'>🤖</h1>
            <h4 style='color: #5a4a6a;'>AI Analysis</h4>
            <p style='font-size: 0.9rem; color: #7b8794;'>MFCC feature extraction & prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card glow-on-hover' style='text-align: center;'>
            <h1 style='color: #a8d5ba;'>🎯</h1>
            <h4 style='color: #5a4a6a;'>Smart Matching</h4>
            <p style='font-size: 0.9rem; color: #7b8794;'>Cosine similarity algorithm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='card glow-on-hover' style='text-align: center;'>
            <h1 style='color: #b39ddb;'>⭐</h1>
            <h4 style='color: #5a4a6a;'>Save Favorites</h4>
            <p style='font-size: 0.9rem; color: #7b8794;'>Build your collection</p>
        </div>
        """, unsafe_allow_html=True)

# Page 2: Upload & Recommend
elif page == "🎧 Upload & Recommend":
    st.markdown("<h1 class='main-title gradient-text'>🎧 Upload & Discover</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload your audio file and let AI find your perfect matches</p>", unsafe_allow_html=True)
    
    # Check API status
    status, models_loaded = check_api_health()
    
    if status != "healthy" or not models_loaded:
        st.error("⚠️ API is not available or models are not loaded. Please check the Home page for status.")
        st.stop()
    
    st.markdown("---")
    
    # Upload Section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #d4a5d4;'>📤 Upload Your Audio</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file (MP3, WAV, M4A)",
        type=['mp3', 'wav', 'm4a'],
        help="Upload a 30-second clip for best results"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # File Info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Filename", uploaded_file.name.split('.')[0][:15] + "...")
        with col2:
            st.metric("📊 Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("🎼 Format", uploaded_file.name.split('.')[-1].upper())
        
        # Audio Preview
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #f5a8c8;'>🎵 Preview</h3>", unsafe_allow_html=True)
        st.audio(uploaded_file)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Settings
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #a8d5ba;'>⚙️ Settings</h3>", unsafe_allow_html=True)
        top_n = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            help="How many similar songs do you want?"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Get Recommendations Button
        if st.button("🔮 Get Recommendations", use_container_width=True):
            with st.spinner("🎵 Analyzing audio and finding matches..."):
                progress_bar = st.progress(0)
                
                # Upload file and get recommendations
                file_bytes = uploaded_file.getvalue()
                
                progress_bar.progress(30)
                time.sleep(0.3)
                
                result = get_recommendations_from_audio(file_bytes, uploaded_file.name, top_n)
                
                progress_bar.progress(100)
                time.sleep(0.3)
                progress_bar.empty()
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success(f"✅ Found {result['count']} recommendations!")
                
                st.markdown("---")
                st.markdown("<h2 class='gradient-text'>🎵 Your Recommendations</h2>", unsafe_allow_html=True)
                
                recommendations = result.get("recommendations", [])
                
                for idx, song in enumerate(recommendations):
                    st.markdown("<div class='song-card fade-in'>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1])
                    
                    with col1:
                        st.markdown(f"<h2 style='color: #d4a5d4; text-align: center;'>{idx + 1}</h2>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"<h3 style='color: #5a4a6a; margin: 0;'>{song['name']}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='margin: 5px 0; color: #7b8794;'>👤 <strong>{song['artists']}</strong></p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='margin: 5px 0; color: #b39ddb;'>💿 {song['album']}</p>", unsafe_allow_html=True)
                    
                    with col3:
                        score = song['similarity_score'] * 100
                        st.markdown(f"<div class='score-badge'>{score:.1f}%</div>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #7b8794;'>Match</p>", unsafe_allow_html=True)
                    
                    with col4:
                        if st.button("⭐ Save", key=f"save_{idx}", use_container_width=True):
                            if song not in st.session_state["favorites"]:
                                st.session_state["favorites"].append(song)
                                st.success("Added to favorites!")
                            else:
                                st.info("Already in favorites!")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='empty-state'>
            <h1>🎵</h1>
            <h3>No audio file uploaded yet</h3>
            <p>Upload an audio file above to get started with AI-powered recommendations</p>
        </div>
        """, unsafe_allow_html=True)

# Page 3: Favorites
elif page == "⭐ Favorites":
    st.markdown("<h1 class='main-title gradient-text'>⭐ Your Favorites</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your saved music collection</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if len(st.session_state["favorites"]) > 0:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<h3 style='color: #d4a5d4;'>💾 {len(st.session_state['favorites'])} songs saved</h3>", unsafe_allow_html=True)
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state["favorites"] = []
                st.rerun()
        
        st.markdown("---")
        
        for idx, song in enumerate(st.session_state["favorites"]):
            st.markdown("<div class='song-card fade-in'>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([0.5, 4, 1])
            
            with col1:
                st.markdown(f"<h2 style='color: #f5a8c8; text-align: center;'>{idx + 1}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<h3 style='color: #5a4a6a; margin: 0;'>{song['name']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin: 5px 0; color: #7b8794;'>👤 <strong>{song['artists']}</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin: 5px 0; color: #b39ddb;'>💿 {song['album']}</p>", unsafe_allow_html=True)
                if 'similarity_score' in song:
                    st.markdown(f"<p style='margin: 5px 0; color: #a8d5ba;'>🎯 Match: {song['similarity_score'] * 100:.1f}%</p>", unsafe_allow_html=True)
            
            with col3:
                if st.button("❌ Remove", key=f"remove_{idx}", use_container_width=True):
                    st.session_state["favorites"].pop(idx)
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='empty-state'>
            <h1>⭐</h1>
            <h3>No favorites yet</h3>
            <p>Start exploring music and save your favorite recommendations!</p>
            <br>
            <p style='font-size: 0.9rem; color: #A259FF;'>Go to Upload & Recommend to discover new music</p>
        </div>
        """, unsafe_allow_html=True)

# Page 4: About
elif page == "ℹ️ About":
    st.markdown("<h1 class='main-title gradient-text'>ℹ️ About SoundMatch AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Discover the technology behind the music</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='card fade-in' style='text-align: center;'>
            <h1 style='font-size: 4rem;'>🎶💜✨</h1>
            <h2 style='color: #b39ddb;'>AI-Powered Music Discovery</h2>
            <p style='font-size: 1.1rem; line-height: 1.8; margin: 25px 0; color: #7b8794;'>
                SoundMatch AI uses cutting-edge machine learning to analyze your audio files 
                and recommend similar tracks from a massive Spotify database.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Section
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>🔬 Technology Stack</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card glow-on-hover'>
            <h3 style='color: #d4a5d4;'>🎵 Audio Processing</h3>
            <ul style='line-height: 2; font-size: 1.05rem; color: #5a4a6a;'>
                <li><strong>MFCC Extraction:</strong> Mel-frequency cepstral coefficients</li>
                <li><strong>Spectral Analysis:</strong> Frequency domain features</li>
                <li><strong>Tempo Detection:</strong> Beat tracking algorithms</li>
                <li><strong>Chroma Features:</strong> Pitch class profiles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card glow-on-hover'>
            <h3 style='color: #f5a8c8;'>🤖 Machine Learning</h3>
            <ul style='line-height: 2; font-size: 1.05rem; color: #5a4a6a;'>
                <li><strong>Hybrid Model:</strong> Stacking ensemble</li>
                <li><strong>Feature Prediction:</strong> Multi-output regression</li>
                <li><strong>Similarity Matching:</strong> Cosine similarity</li>
                <li><strong>Deep Learning:</strong> Neural network layers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How It Works Section
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>⚙️ How It Works</h2>", unsafe_allow_html=True)
    
    steps = [
        ("1️⃣", "Upload Audio", "You upload an audio file (MP3, WAV, or M4A)", "#d4a5d4"),
        ("2️⃣", "Feature Extraction", "MFCC and spectral features are extracted", "#f5a8c8"),
        ("3️⃣", "AI Prediction", "Hybrid model predicts music attributes", "#a8d5ba"),
        ("4️⃣", "Similarity Search", "Cosine similarity finds matching tracks", "#d4a5d4"),
        ("5️⃣", "Get Results", "Receive personalized recommendations", "#f5a8c8")
    ]
    
    for emoji, title, description, color in steps:
        st.markdown(f"""
        <div class='card glow-on-hover' style='border-left: 4px solid {color};'>
            <h2 style='color: {color}; margin: 0;'>{emoji} {title}</h2>
            <p style='font-size: 1.1rem; margin-top: 10px; color: #7b8794;'>{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Section
    st.markdown("<h2 class='gradient-text' style='text-align: center;'>✨ Predicted Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h3 style='color: #d4a5d4;'>🎵 Audio Attributes</h3>
            <ul style='text-align: left; line-height: 2; color: #5a4a6a;'>
                <li>Danceability</li>
                <li>Energy</li>
                <li>Loudness</li>
                <li>Valence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h3 style='color: #f5a8c8;'>🎹 Musical Properties</h3>
            <ul style='text-align: left; line-height: 2; color: #5a4a6a;'>
                <li>Key</li>
                <li>Mode</li>
                <li>Tempo</li>
                <li>Time Signature</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h3 style='color: #a8d5ba;'>🎤 Content Analysis</h3>
            <ul style='text-align: left; line-height: 2; color: #5a4a6a;'>
                <li>Speechiness</li>
                <li>Acousticness</li>
                <li>Instrumentalness</li>
                <li>Liveness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <h3 style='color: #b39ddb;'>💜 Built with Love & AI</h3>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #7b8794;'>
            SoundMatch AI combines the power of machine learning, audio signal processing, 
            and massive datasets to help you discover your next favorite song.
        </p>
        <br>
        <p style='color: #d4a5d4; font-size: 1.2rem;'>
            🎵 Start exploring music today! 🎵
        </p>
    </div>
    """, unsafe_allow_html=True)