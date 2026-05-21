from sentence_transformers import SentenceTransformer, util
import re

# Load multilingual model (supports Kannada)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def lexical_diversity(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)

def avg_sentence_length(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0
    return sum(len(s.split()) for s in sentences) / len(sentences)
def information_loss(original, summary):
    orig_words = set(original.split())
    sum_words = set(summary.split())

    missing = len(orig_words - sum_words)
    added = len(sum_words - orig_words)

    return missing, added