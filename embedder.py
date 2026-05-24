# embedder.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# 한국어/영어 동시 지원, 로컬 실행, 비용 0

def semantic_dedup(news_list: list, threshold: float = 0.80) -> list:
    """
    임베딩 기반 의미론적 중복 제거
    같은 사건을 다룬 기사 → 대표 1개만 선택
    """
    if len(news_list) <= 1:
        return news_list

    titles = [n.get('title', '') for n in news_list]
    embeddings = model.encode(titles, show_progress_bar=False)

    # 코사인 유사도 행렬
    sim_matrix = cosine_similarity(embeddings)

    # 대표 기사 선택 (greedy)
    selected = []
    excluded = set()

    for i in range(len(news_list)):
        if i in excluded:
            continue
        selected.append(news_list[i])

        # 유사도 threshold 이상인 것들 제외
        for j in range(i + 1, len(news_list)):
            if sim_matrix[i][j] >= threshold:
                excluded.add(j)

    return selected