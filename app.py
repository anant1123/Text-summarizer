import streamlit as st
from PyPDF2 import PdfReader
import spacy
from goose3 import Goose
import regex  # Import the 'regex' library instead of 're'

# Load the spaCy model with 'sentencizer'
with open("en_core_web_lg.pkl", "rb") as f:
    nlp = pickle.load(f)

sentencizer = nlp.add_pipe("sentencizer")
def scrape_text_from_url(url):
    try:
        g = Goose()
        article = g.extract(url=url)
        text = article.cleaned_text
        return text
    except Exception as e:
        return f"Error extracting text from URL: {str(e)}"

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def preprocess_text(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def extractive_summarization(sentences, num_sentences=3):
    sentence_scores = [(i, sum(token.is_alpha for token in nlp(sentence))) for i, sentence in enumerate(sentences)]
    summary_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    summary = ' '.join(sentences[i] for i, _ in summary_sentences)
    return summary

def main():
    st.title("Text Extraction and Summarization App")

    choice = st.radio("Choose an option:", ["Wikipedia Link", "PDF File", "Copy and Paste Text"])

    if choice == "Wikipedia Link":
        url = st.text_input("Wikipedia Link:")
        text = scrape_text_from_url(url)
    elif choice == "PDF File":
        pdf_file = st.file_uploader("Upload PDF File:", type=["pdf"])
        if pdf_file is not None:
            text = extract_text_from_pdf(pdf_file)
    elif choice == "Copy and Paste Text":
        text = st.text_area("Paste Text:")

    if st.button("Extract and Summarize"):
        sentences = preprocess_text(text)
        summary = extractive_summarization(sentences)
        cleaned_summary = regex.sub(r'\[\d+\]', '', summary)  # Using 'regex' instead of 're'

        original_text_length = len(text)
        summarized_text_length = len(cleaned_summary)

        st.subheader("Extraction Result")
        st.write(cleaned_summary)
        st.write(f"Original Text Length: {original_text_length} characters")
        st.write(f"Summarized Text Length: {summarized_text_length} characters")

if __name__ == "__main__":
    main()
