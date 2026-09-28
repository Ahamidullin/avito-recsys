"""кодирование текстов и точный поиск ближайших соседей. индекс не нужен тк корпус в 189k влезает в память целиком"""
import numpy as np
import torch
def encode(model, texts, batch_size=64, prefix=""):
    """эмебеды нормированы поэтому скалярное произведение это косинус"""
    texts = [prefix + t for t in texts]
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return emb.astype(np.float32)


def topk(query_emb, corpus_emb, k=50, device="mps", chunk=256):
    """для каждого запроса индексы k ближайших объявлений. Считаем кусками, чтобы не забить память"""
    corpus = torch.from_numpy(corpus_emb).to(device)
    out = []
    for start in range(0, len(query_emb), chunk):
        q = torch.from_numpy(query_emb[start:start + chunk]).to(device)
        scores = q @ corpus.T
        idx = scores.topk(k, dim=1).indices.cpu().numpy()
        out.append(idx)
    return np.concatenate(out)
