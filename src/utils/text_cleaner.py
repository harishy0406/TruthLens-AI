import re
import spacy

nlp = spacy.load("en_core_web_sm")
def clean_text(text):

    text = str(text)

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove html tags
    text = re.sub(r"<.*?>", "", text)

    # remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def lemmatize_text(text):

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(tokens)