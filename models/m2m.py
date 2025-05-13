
# 109th March 2023

import time
from transformers import (
   M2M100ForConditionalGeneration,
   M2M100Tokenizer
)
import os

import re
import nltk

from nltk.util import ngrams

#Download on your own device
nltk.data.path.append(r"D:\Major_Project\NEWUI/.venv/Lib/site-packages/nltk/nltk_data")
nltk.download('punkt_tab', download_dir=r"D:\Major_Project\NEWUI/.venv/Lib/site-packages/nltk/nltk_data")
# nltk.download('punkt')
from datetime import datetime
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# nltk.download('punkt')  # Ensure NLTK punkt tokenizer is downloaded

import Levenshtein
from Levenshtein import ratio

generated_wishes = {}

def generate_title(data, tokenizer, model, max_retries=50):
   global generated_wishes

   if data not in generated_wishes:
      generated_wishes[data] = []  # Store wishes specific to the prompt

   inputs = tokenizer(data, return_tensors="pt")
   inputs = {k: v.to(model.device) for k, v in inputs.items()}

   def is_too_similar(new_text, generatedwishes):
      """Check if the new wish is too similar to previously generated ones."""
      return any(ratio(new_text, old_text) >= 0.90 for old_text in generatedwishes[data])

   for i in range(max_retries):

      outputs = model.generate(
         **inputs,
         do_sample=True,
         top_k=100,
         top_p=0.9,
         temperature=1.2,
         repetition_penalty=1.2,
         num_return_sequences=1
      )
      generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

      if not is_too_similar(generated_text, generated_wishes):
         generated_wishes[data].append(generated_text)
         return generated_text

   return generated_text

festival_dict = {
   "Dashain": ["dashain", "bada dashain", "bijaya dashami", "dashami"],
   "Tihar": ["tihar", "deepawali", "deewali", "diwali"],
   "Chhath": ["chhath", "surya sasti", "suryasasthi", "surya shashthi", "chhathimaiya", "chhathparba", "chhathparva", "chhathpuja", "chhaith", "dala chhath", "dala puja"],
   "New Year": ["naya barsa", "nawa barsa", "nava barsa"],
   "Holi": ["fagu purnima", "holi", "phagu purnima"]
}


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
   # print(detected_festival)

   # if detected_festival:
   #    corrected_text = prompt
   #    for festival, variations in festival_dict.items():
   #          for variation in variations:
   #             corrected_text = re.sub(r'\b{}\b'.format(re.escape(variation)), detected_festival, corrected_text, flags=re.IGNORECASE)
   #    return corrected_text
   # else:
   #    return None
   
   if detected_festival:
      new_prompt = f"For the festival {detected_festival}, {prompt} for {detected_festival}"
      return new_prompt, detected_festival
   else:
      return None, None


def _extract_year_from_wish(generated_text):
   match = re.search(r'([०१२३४५६७८९]{4})', generated_text)
   if match:
      return match.group()
   # else:
   #    current_year = datetime.now().year + 57  # Add 57 for Nepali year
   #    nepali_current_year = ''.join('०१२३४५६७८९'[int(digit)] for digit in str(current_year))
      
   #    return nepali_current_year


def check_and_append_year(prompt):
   # Pattern to find a four-digit year
   year_pattern = r'\b(\d{4})\b'
   match = re.search(year_pattern, prompt)

   if match:
      return prompt

   # Convert the current year to Nepali numerals
   current_year = datetime.now().year + 57 - 1
   nepali_current_year = ''.join('०१२३४५६७८९'[int(digit)] for digit in str(current_year))

   return nepali_current_year


def extract_year_from_prompt(prompt):
   updated_prompt = check_and_append_year(prompt)
   year_pattern = r'\b(\d{4})\b'
   match = re.search(year_pattern, updated_prompt)
   if match:
      year = int(match.group(1))
      nepali_year = ''.join('०१२३४५६७८९'[int(digit)] for digit in str(year))
      return nepali_year
   return None


def _year_corrected(generated_text, prompt):
   generated_year_nepali = _extract_year_from_wish(generated_text)
   input_year_nepali = extract_year_from_prompt(prompt)

   if generated_year_nepali and input_year_nepali:
      if generated_year_nepali != input_year_nepali:
            updated_generated_text = re.sub(r'([०१२३४५६७८९]{4})', input_year_nepali, generated_text)
            return updated_generated_text
      else:
            return generated_text
   else:
      return generated_text + ' ' + input_year_nepali[1] if input_year_nepali[1] else generated_text


def generate_text(prompt, model, tokenizer):
   tokenizer.src_lang = "en"
   # prompt_with_timestamp = f"{prompt} {int(time.time())}"  # Add a timestamp to make it unique each time

   
   prompt_words = nltk.word_tokenize(prompt.lower())  # Tokenize and convert to lowercase
   bigrams = list(ngrams(prompt_words, 2))
   
   for word in prompt_words:
      for festival, variations in festival_dict.items():
            # Check if the word matches the festival name itself (key) or any of its variations (values)
            if word == festival.lower() or word in [var.lower() for var in variations]:
               # encoded_hi = tokenizer(prompt, return_tensors="pt")
               # generated_tokens = model.generate(**encoded_hi, num_return_sequences=2,forced_bos_token_id=tokenizer.get_lang_id("ne"))
               # generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
               generated_text = generate_title(prompt, tokenizer, model)
               year_corrected_wish = _year_corrected(generated_text, prompt)
               return year_corrected_wish, festival.lower()
   
   for bigram in bigrams:
         bigram_str = ' '.join(bigram)  # Convert bigram tuple to string
         for festival, variations in festival_dict.items():
            if bigram_str == festival.lower() or bigram_str in [var.lower() for var in variations]:
               # encoded_hi = tokenizer(prompt, return_tensors="pt")
               # generated_tokens = model.generate(
               #    **encoded_hi, num_return_sequences=2,
               #    forced_bos_token_id=tokenizer.get_lang_id("ne")
               # )
               # generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
               generated_text = generate_title(prompt, tokenizer, model)
               year_corrected_wish = _year_corrected(generated_text, prompt)
               return year_corrected_wish, festival.lower()  # Return on first bigram match

   corrected_prompt, festival = correct_festival_name(prompt)
   if corrected_prompt:
      # encoded_hi = tokenizer(corrected_prompt, return_tensors="pt")
      # generated_tokens = model.generate(**encoded_hi, num_return_sequences=2,forced_bos_token_id=tokenizer.get_lang_id("ne"))
      # generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
      generated_text = generate_title(prompt, tokenizer, model)
      year_corrected_wish = _year_corrected(generated_text, corrected_prompt)
      return year_corrected_wish, festival.lower()
   else:
      return None, None                              
   
   
   # corrected_prompt = correct_festival_name(prompt)
   
   # if corrected_prompt==None:
   #    return None
   # else:
   #    encoded_hi = tokenizer(corrected_prompt, return_tensors="pt")
   #    generated_tokens = model.generate(**encoded_hi, num_return_sequences=2,forced_bos_token_id=tokenizer.get_lang_id("ne"))
   #    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
   #    year_corrected_wish = generated_text(generated_text, corrected_prompt)
   #    return year_corrected_wish
   