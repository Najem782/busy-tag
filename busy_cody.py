import streamlit as st
from PIL import Image
import os

# Function to create the busy tag
def create_busy_tag(image_path):
    # Load image
    try:
        image = Image.open(image_path)
        image = image.resize((60, 60))  # Resize image to fit the tag size
        st.image(image, caption="Busy", use_column_width=False)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Streamlit UI for selecting an image
st.title("Select Your Busy Image!")
st.write("Choose a cute image or GIF to represent your 'busy' status:")

# List of images and GIFs in the "images/" folder
image_folder = "images"
images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Let user choose an image or GIF
image_choice = st.selectbox("Select Image/GIF", images)

# Create the busy tag with the selected image
image_path = os.path.join(image_folder, image_choice)
create_busy_tag(image_path)
