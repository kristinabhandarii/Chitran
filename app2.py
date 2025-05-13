
#10th March 2023

import streamlit as st
from models.m2m import generate_text
from models.ldm import generate_image
from models.textstyle import generate_styled_image
import requests
import torch
import os
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from PIL import Image
import io

# Set environment variable to disable file watcher if needed
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"


# Function to load the model and tokenizer for M2M
@st.cache_resource

def load_m2m():
    model_dir = os.path.join(os.path.dirname(__file__), "models", "checkpoints", "m2mcheckpoint")
    model = M2M100ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = M2M100Tokenizer.from_pretrained(model_dir, src_lang="en", tgt_lang="ne")
    tokenizer.src_lang = "en"
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_m2m()
detected_festival = None

# Streamlit app main function
def main():
    # Initialize session state if not already present
    if 'page' not in st.session_state:
        st.session_state.page = "Text Generation"  # Start with the Text Generation page
    
    # Based on current page, call the relevant function
    if st.session_state.page == "Text Generation":
        text_generation_page(model, tokenizer)
    elif st.session_state.page == "Image Generation":
        image_generation_page()
    elif st.session_state.page == "Style Generation":
        text_styling_page()


# Page 1: Text Generation
def text_generation_page(model, tokenizer):
    st.title("Text Generation")
    global detected_festival
        
    english_prompt = st.text_area("Enter your English prompt for the festivals Dashain, Tihar, Chhath, Holi or New Year:", key="english_prompt_input")
    
    if st.button("Generate Nepali Wish"):
        if english_prompt:
            nepali_wish, detected_festival = generate_text(english_prompt, model, tokenizer)
            if nepali_wish:
                st.write("Generated Nepali Wish:")
                st.write(nepali_wish)
                st.session_state.nepali_wish = nepali_wish  # Store the generated wish for later use
                st.session_state.festival = detected_festival
                st.success("Text Generated! Go to Image Generation")
            elif nepali_wish==None:
                st.warning("Chitran only supports 5 festivals: Dashain, Tihar, Chhath, New Year and Holi.")
        else:
            st.warning("Please enter a valid prompt.")
        
        if st.button("Rerun Text Generation"):
            del st.session_state.nepali_wish  # Remove stored text
            st.session_state.page = "Text Generation"
            nepali_wish, detected_festival = generate_text(english_prompt, model, tokenizer)
            if nepali_wish:
                st.write("Generated Nepali Wish:")
                st.write(nepali_wish)
                st.session_state.nepali_wish = nepali_wish  # Store the generated wish for later use
                st.session_state.festival = detected_festival
                st.success("Text Generated! Click Next to go to Image Generation.")
            # st.rerun()  # Rerun to refresh UI
            elif nepali_wish==None:
                st.warning("Chitran only supports 5 festivals: Dashain, Tihar, Chhath, New Year and Holi.")
            st.rerun()
    
    if st.button("Go To Image Generation"):
        st.session_state.page = "Image Generation"
        st.rerun()


# Page 2: Image Generation
def image_generation_page():
    st.title("Image Generation")
    
    if "nepali_wish" not in st.session_state:
        st.warning("Please generate text first before proceeding.")
        return
    
    image_prompt = st.text_area("Enter image description for the festivals Dashain, Tihar, Chhath, New Year or Holi:", key="image_prompt_input")
    
    if st.button("Generate Image"):
        # global detected_festival
        if image_prompt:
            generated_image, detected_festival2  = generate_image(image_prompt)
            if generated_image:
                # st.write("Generated Festive Image")
                st.image(generated_image,  use_container_width=False)
                st.session_state.generated_image = generated_image  # Store the generated image
                st.success("Image Generated! Go to Text Styling")
            elif generated_image == None:
                st.warning("Chitran only supports 5 festivals: Dashain, Tihar, Chhath, New Year and Holi.")
        else:
            st.warning("Please enter a valid description.")
        
        if st.button("Rerun Image Generation"):
            del st.session_state.generated_image
            st.session_state.page = "Image Generation"
            if generated_image:
                st.write("Generated Festive Image")
                st.image(generated_image, caption="Generated Image", use_container_width=True)
                st.session_state.generated_image = generated_image  # Store the generated image
                st.success("Image Generated! Go to Text Styling")
            elif generated_image == None:
                st.warning("Chitran only supports 5 festivals: Dashain, Tihar, Chhath, New Year and Holi.")

            img_buffer = io.BytesIO()
            generated_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            st.download_button(
                label="Download Styled Image",
                data=img_buffer,
                file_name="styled_image.png",
                mime="image/png"
            )
    if st.button("Go To Text Styling"):
        st.session_state.page = "Style Generation"
        st.rerun()

# Page 3: Text Styling and Integration
def text_styling_page():
    st.title("Poster Generation")
    global detected_festival
    
    if "nepali_wish" not in st.session_state or "generated_image" not in st.session_state:
        st.warning("Please generate text and image first before proceeding.")
        return
    
    # Combine the generated text with the image
    if st.button("Generate Styled Image"):
        # global detected_festival
        styled_image = generate_styled_image(st.session_state.nepali_wish, st.session_state.generated_image, st.session_state.festival )
        st.image(styled_image, caption="Styled Image", use_container_width=True)
        st.success("Styled Image Generated!")
        
        # Prepare image for download
        img_buffer = io.BytesIO()
        styled_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        st.download_button(
            label="Download Styled Image",
            data=img_buffer,
            file_name="styled_image.png",
            mime="image/png"
        )

# Run the Streamlit app
if __name__ == "__main__":
    main()