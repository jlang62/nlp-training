import json
import spacy
from textblob import TextBlob
from transformers import pipeline
import pdf2image
import pytesseract

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load summarization model
summarizer = pipeline("summarization")

# Load zero-shot classifier model for topic detection
classifier = pipeline("zero-shot-classification")

def extract_text_from_pdf(file_path):
    # Code for extracting text from PDF
    # Using pdf2image and pytesseract
    images = pdf2image.convert_from_path(file_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extract_text_from_image(file_path):
    # Extract text from an image file
    text = pytesseract.image_to_string(file_path)
    return text

def extract_entities(text):
    # Extract named entities
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
    # Get sentiment score using TextBlob
    blob = TextBlob(text)
    return blob.sentiment.polarity

def extract_keywords(text):
    # Extract keywords (e.g., using noun chunks)
    doc = nlp(text)
    keywords = [chunk.text.strip() for chunk in doc.noun_chunks]
    
    # Filter out non-informative words (e.g., articles, prepositions)
    filtered_keywords = [word for word in keywords if len(word.split()) > 1]  # Avoid single words
    return filtered_keywords

def summarize_text(text, max_length=130, min_length=30):
    # Summarize text using a pre-trained model
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def extract_topic(text):
    # Use zero-shot classification to detect the topic of the document
    candidate_labels = ["Law", "Litigation", "Technology", "Business", "Science", "Health"]
    result = classifier(text, candidate_labels)
    return result['labels'][0]  # Return the most likely topic

def process_file(file_path, file_type='pdf'):
    # Extract text
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'image':
        text = extract_text_from_image(file_path)
    else:
        print("Unsupported file type.")
        return

    # Display extracted text (optional)
    print("Extracted Text:\n", text)

    # Analyze entities
    entities = extract_entities(text)
    print("\nExtracted Entities:")
    for entity_type, values in entities.items():
        print(f"{entity_type.capitalize()}: {values}")

    # Sentiment Analysis
    sentiment_score = get_sentiment(text)
    print("\nSentiment Score:", sentiment_score)

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
        "keywords": keywords,
        "summary": summary,
        "topic": topic
    }

    # Print or store JSON
    json_output = json.dumps(output_data, indent=4)
    print("\nJSON Output:", json_output)
    return output_data
