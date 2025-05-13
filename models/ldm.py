
#10th March 2023

from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
import os 
import requests
from PIL import Image
import io
import re
import nltk
from nltk.util import ngrams
import time
import Levenshtein
from Levenshtein import ratio


festival_dict = {
    "Dashain": ["dashain", "bada dashain", "bijaya dashami", "dasai", "dashai", "dashami"],
    "Tihar": ["tihar", "deepawali", "deewali", "diwali"],
    "Chhath": ["chhath", "surya sasti", "suryasasthi", "surya shashthi", "chhathimaiya", "chhathparba", "chhathparva", "chhathpuja", "chhaith", "dala chhath", "dala puja"],
    "New Year": ["naya barsa", "nawa barsa", "nava barsa"],
    "Holi": ["fagu purnima", "holi", "phagu purnima"]
}


model_dir = "checkpoints/ldmcheckpoint\sd.pth"

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def calculate_levenshtein_similarity(prompt_tokens, variation_tokens_combined):
    # Convert token sets to sorted lists for string comparison
    prompt_str = ' '.join(sorted(map(str, prompt_tokens)))
    variation_str = ' '.join(sorted(map(str, variation_tokens_combined)))
    
    return ratio(prompt_str, variation_str)


def check_festival_name(prompt):
    # text_tokens = set(nltk.word_tokenize(prompt.lower()))
    text_tokens = set(prompt.lower().split())
    
    unigram_tokens = set(text_tokens)
    bigram_tokens = set([' '.join(bigram) for bigram in ngrams(text_tokens, 2)])
    
    prompt_tokens = unigram_tokens.union(bigram_tokens)
    
    best_match = None
    highest_similarity = 0

    for festival, variations in festival_dict.items():
        for variation in variations:
                # variation_tokens = set(nltk.word_tokenize(variation.lower()))
                # variation_tokens = set(variation.lower().split())  # Use split() instead of word_tokenize()
                # unigram_variation_tokens = set(variation_tokens)
                # bigram_variation_tokens = set(ngrams(variation_tokens, 2))
                # variation_tokens_combined = unigram_variation_tokens.union(bigram_variation_tokens)
                variation_lower = variation.lower()
                
                for prompt_token in prompt_tokens:
                    jaccard_similarity = calculate_jaccard_similarity(prompt_tokens, variation_lower)
                    levenshtein_similarity = max(calculate_levenshtein_similarity(prompt_token, var_token) for var_token in variation_lower.split())
                    similarity = max(jaccard_similarity, levenshtein_similarity)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = festival

    if highest_similarity >= 0.6:
        return best_match
    else:
        return None


def correct_festival_name(prompt):
    input_text_lower = prompt.lower()
    detected_festival = check_festival_name(input_text_lower)
    
    if detected_festival:
        return f"For the festival {detected_festival}, {prompt} for {detected_festival}", detected_festival
    else:
        return None, None


def api_call(user_input):
    url = "https://0dd8-34-83-140-188.ngrok-free.app/predict"
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json={"prompt": user_input}, headers=headers)

    if response.status_code == 200:
        # Save the image to a file
        with open("generated_image.png", "wb") as f:
            f.write(response.content)
        # st.success("Image saved as generated_image.png")

        # Display the image in Streamlit
        generated_image = Image.open("generated_image.png")
    
    return generated_image


def generate_image(user_input):
    prompt_with_timestamp = f"{user_input} {int(time.time())}"
    
    prompt_words = nltk.word_tokenize(prompt_with_timestamp.lower())  # Tokenize and convert to lowercase
    bigrams = list(ngrams(prompt_words, 2))
    
    # matched_festivals = []
    
    for word in prompt_words:
        for festival, variations in festival_dict.items():
            if word == festival.lower() or word in [var.lower() for var in variations]:
                generated_image = api_call(user_input)
                return generated_image, festival.lower()
    
    for bigram in bigrams:
        bigram_str = ' '.join(bigram)
        for festival, variations in festival_dict.items():
            if bigram_str == festival.lower() or bigram_str in [var.lower() for var in variations]:
                generated_image = api_call(user_input)
                return generated_image, festival.lower()

    corrected_prompt, festival = correct_festival_name(user_input)
    if corrected_prompt:
        generated_image = api_call(corrected_prompt)
        return generated_image, None
    return None, festival