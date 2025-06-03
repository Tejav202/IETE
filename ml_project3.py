import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Computer Vision Playground",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 4px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .image-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .title {
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #34495e;
        font-size: 1.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_image(image_file):
    """Load image from file"""
    if image_file is not None:
        image = Image.open(image_file)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return None

def detect_edges(image, threshold1=100, threshold2=200):
    """Detect edges using Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def detect_contours(image, min_area=100):
    """Detect and draw contours"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Draw contours
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result

def analyze_colors(image, num_colors=5):
    """Analyze dominant colors in the image"""
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    return centers

def apply_morphological_operations(image, operation='dilate', kernel_size=5):
    """Apply morphological operations"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'dilate':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erode':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def main():
    st.markdown('<div class="title">👁️ Computer Vision Playground</div>', unsafe_allow_html=True)
    st.write("Explore basic computer vision operations using OpenCV!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image
        image = load_image(uploaded_file)
        if image is not None:
            # Display original image
            st.markdown('<div class="subtitle">Original Image</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Create columns for options
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Edge Detection")
                edge_detection = st.checkbox("Detect Edges")
                if edge_detection:
                    threshold1 = st.slider("Threshold 1", 0, 255, 100)
                    threshold2 = st.slider("Threshold 2", 0, 255, 200)
                
                st.subheader("Contour Detection")
                contour_detection = st.checkbox("Detect Contours")
                if contour_detection:
                    min_area = st.slider("Minimum Contour Area", 0, 1000, 100)
            
            with col2:
                st.subheader("Morphological Operations")
                morph_op = st.selectbox(
                    "Select Operation",
                    ["None", "Dilate", "Erode", "Open", "Close"]
                )
                if morph_op != "None":
                    kernel_size = st.slider("Kernel Size", 3, 15, 5, 2)
                
                st.subheader("Color Analysis")
                color_analysis = st.checkbox("Analyze Colors")
                if color_analysis:
                    num_colors = st.slider("Number of Colors", 2, 10, 5)
            
            # Process image
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    # Create a copy of the original image
                    processed_image = image.copy()
                    
                    # Apply selected operations
                    if edge_detection:
                        processed_image = detect_edges(processed_image, threshold1, threshold2)
                    
                    if contour_detection:
                        processed_image = detect_contours(processed_image, min_area)
                    
                    if morph_op != "None":
                        processed_image = apply_morphological_operations(
                            processed_image,
                            morph_op.lower(),
                            kernel_size
                        )
                    
                    # Display processed image
                    st.markdown('<div class="subtitle">Processed Image</div>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Color analysis
                    if color_analysis:
                        st.markdown('<div class="subtitle">Color Analysis</div>', unsafe_allow_html=True)
                        colors = analyze_colors(image, num_colors)
                        
                        # Display color palette
                        color_palette = np.zeros((100, 500, 3), dtype=np.uint8)
                        color_width = 500 // num_colors
                        for i, color in enumerate(colors):
                            color_palette[:, i*color_width:(i+1)*color_width] = color
                        
                        st.image(color_palette, use_column_width=True)
                        
                        # Display RGB values
                        st.write("Dominant Colors (RGB):")
                        for i, color in enumerate(colors):
                            st.write(f"Color {i+1}: RGB{tuple(color)}")
                    
                    # Download option
                    if st.button("Download Processed Image"):
                        # Convert to PIL Image
                        pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                        # Save to bytes
                        img_byte_arr = io.BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        # Create download button
                        st.download_button(
                            label="Download Image",
                            data=img_byte_arr,
                            file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )

    # Add information
    st.sidebar.header("About")
    st.sidebar.info("""
    This application demonstrates basic computer vision operations using OpenCV:
    
    Features:
    - Edge Detection (Canny)
    - Contour Detection
    - Morphological Operations
    - Color Analysis
    
    Upload an image and try different operations!
    """)
    
    # Add usage instructions
    st.sidebar.header("How to Use")
    st.sidebar.markdown("""
    1. Upload an image
    2. Select desired operations
    3. Adjust parameters
    4. Click 'Process Image'
    5. Download the result
    
    Note: You can combine multiple operations to see their effects.
    """)

if __name__ == "__main__":
    main() 