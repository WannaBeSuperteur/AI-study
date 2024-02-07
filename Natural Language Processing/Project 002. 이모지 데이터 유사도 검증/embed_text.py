from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

# fastest model from https://www.sbert.net/docs/pretrained_models.html
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')


def embed_text(text):
    return sbert_model.encode(text)


def compute_cos_sim(text1, text2):
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    
    cos_sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return cos_sim


if __name__ == '__main__':
    sentence1 = 'Natural Language Processing gives machine a life.'
    sentence2 = 'Natural Language Processing gives computer a life.'

    print(embed_text(sentence1))
    print(embed_text(sentence2))
    print(compute_cos_sim(sentence1, sentence2))
