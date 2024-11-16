import json
import spacy
from textblob import TextBlob
from transformers import pipeline
import pdf2image
import pytesseract
import os

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load summarization model
summarizer = pipeline("summarization")

# Load zero-shot classifier model for topic detection
classifier = pipeline("zero-shot-classification")

def extract_text_from_pdf(file_path):
    """
    Convert PDF to images and extract text using pytesseract
    """
    try:
        images = pdf2image.convert_from_path(file_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(file_path):
    """
    Extract text from an image file using pytesseract
    """
    try:
        text = pytesseract.image_to_string(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_entities(text):
    """
    Extract named entities from the text using SpaCy
    """
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)

    # Remove redundant entities (e.g., multiple mentions of the same person)
    for entity_type, values in entities.items():
        entities[entity_type] = list(set(values))  # Remove duplicates
    return entities

def get_sentiment(text):
    """
    Get sentiment polarity using TextBlob
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment_label = "POSITIVE" if sentiment_score > 0 else "NEGATIVE" if sentiment_score < 0 else "NEUTRAL"
    return sentiment_score, sentiment_label

def extract_keywords(text):
    """
    Extract keywords using SpaCy's noun chunks
    """
    doc = nlp(text)
    keywords = [chunk.text.strip() for chunk in doc.noun_chunks]
    
    # Filter out non-informative words (e.g., articles, prepositions)
    filtered_keywords = [word for word in keywords if len(word.split()) > 1]  # Avoid single words
    return filtered_keywords

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarize the text using a pre-trained model
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summary not available."

def extract_topic(text):
    """
    Use zero-shot classification to detect the topic of the document
    """
    candidate_labels = ["Law", "Litigation", "Technology", "Business", "Science", "Health"]
    result = classifier(text, candidate_labels)
    return result['labels'][0]  # Return the most likely topic

def process_file(file_path, file_type='pdf'):
    """
    Main function to process the file and return a detailed output
    """
    # Extract text from the file based on type
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'image':
        text = extract_text_from_image(file_path)
    else:
        print("Unsupported file type.")
        return {}

    # Ensure the extracted text is valid
    if not isinstance(text, str) or len(text.strip()) == 0:
        print("No valid text extracted.")
        return {}

    # Display extracted text (optional)
    print("Extracted Text:\n", text)

    # Analyze entities
    entities = extract_entities(text)
    print("\nExtracted Entities:")
    for entity_type, values in entities.items():
        print(f"{entity_type.capitalize()}: {values}")

    # Sentiment Analysis
    sentiment_score, sentiment_label = get_sentiment(text)
    print("\nSentiment Score:", sentiment_score, "Sentiment Label:", sentiment_label)

    # Keyword Extraction
    keywords = extract_keywords(text)
    print("\nKeywords:", keywords)

    # Summarization
    summary = summarize_text(text)
    print("\nSummary:", summary)

    # Topic Detection
    topic = extract_topic(text)
    print("\nDetected Topic:", topic)

    # Output everything as JSON
    output_data = {
        "extracted_text": text,
        "entities": entities,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "keywords": keywords,
        "summary": summary,
        "topic": topic
    }

    # Print or store JSON
    json_output = json.dumps(output_data, indent=4)
    print("\nJSON Output:", json_output)
    
    return output_data

