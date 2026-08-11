# 检索抽象层：Retriever 基类 + 零依赖 BM25 默认实现。
# 中文字符二元分词 + 倒排索引，同一 query 结果确定可复现；
# 后续接入 embedding/向量后端时只需新增 Retriever 实现，调用方无需改动。
import math
import re

from knowledge.tcm_knowledge import get_corpus

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """中文按字符二元分词（长度 1 时保留单字），ASCII 字母数字按整词。"""
    tokens = []
    for chunk in _CJK_RE.findall(text or ""):
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    tokens.extend(_ASCII_RE.findall((text or "").lower()))
    return tokens


class Retriever:
    """检索接口：search(query, top_k) 返回命中列表，具体实现决定打分策略。"""

    def search(self, query, top_k=5):
        raise NotImplementedError


class BM25Retriever(Retriever):
    """零依赖轻量 BM25：构造时一次性建倒排索引，检索纯查表无随机性。"""

    def __init__(self, entries=None, k1=1.5, b=0.75):
        self._k1 = k1
        self._b = b
        self._entries = list(entries) if entries is not None else get_corpus()
        self._doc_tokens = []
        self._doc_freqs = []
        self._df = {}
        total_len = 0
        for entry in self._entries:
            # 检索面：标题 + 标签 + 正文，三者共同构成该语料的检索文本
            text = " ".join([entry.get("title", "")] + list(entry.get("tags", [])) + [entry.get("content", "")])
            tokens = tokenize(text)
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self._doc_tokens.append(tokens)
            self._doc_freqs.append(freq)
            total_len += len(tokens)
            for token in freq:
                self._df[token] = self._df.get(token, 0) + 1
        self._doc_count = len(self._entries)
        self._avg_len = total_len / self._doc_count if self._doc_count else 0.0

    def search(self, query, top_k=5):
        query = (query or "").strip()
        if not query or not self._doc_count:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        scores = []
        for index, freq in enumerate(self._doc_freqs):
            score = 0.0
            doc_len = len(self._doc_tokens[index])
            for token in query_tokens:
                tf = freq.get(token, 0)
                if not tf:
                    continue
                df = self._df.get(token, 0)
                idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_len)
                score += idf * tf * (self._k1 + 1) / denom
            if score > 0:
                scores.append((score, index))

        # 分数降序，同分按 id 排序保证结果稳定可复现
        scores.sort(key=lambda item: (-item[0], self._entries[item[1]]["id"]))
        top_k = max(1, min(int(top_k), 20))
        hits = []
        for score, index in scores[:top_k]:
            entry = self._entries[index]
            hits.append(
                {
                    "id": entry["id"],
                    "title": entry["title"],
                    "category": entry.get("category", ""),
                    "score": round(score, 3),
                    "source": entry.get("source_file", ""),
                    # content 为自包含文本段，透传给下游直接引用（RAG 注入回复）
                    "content": entry.get("content", ""),
                }
            )
        return hits


# 默认全局检索器：模块加载时建索引，同 query 结果确定可复现
DEFAULT_RETRIEVER = BM25Retriever()


class TFIDFRetriever(Retriever):
    """零依赖轻量 TF-IDF：索引面与 BM25 完全一致（同 tokenizer、标题+标签+正文），
    仅打分策略不同：sublinear TF（1+log tf）× IDF，文档/查询向量 L2 归一化后余弦相似度。
    作为 BM25 的对比后端（及后续远程 embedding 的前置参照），
    质量对比由 evaluation/compare_retrievers.py 输出，确定性可复现。
    """

    def __init__(self, entries=None):
        self._entries = list(entries) if entries is not None else get_corpus()
        self._doc_freqs = []
        self._df = {}
        for entry in self._entries:
            # 检索面与 BM25 保持一致，保证对比的唯一变量是打分策略
            text = " ".join([entry.get("title", "")] + list(entry.get("tags", [])) + [entry.get("content", "")])
            freq = {}
            for token in tokenize(text):
                freq[token] = freq.get(token, 0) + 1
            self._doc_freqs.append(freq)
            for token in freq:
                self._df[token] = self._df.get(token, 0) + 1
        self._doc_count = len(self._entries)
        # 文档权重向量预计算（检索时纯查表）：sublinear TF × IDF，L2 归一化
        self._doc_weights = []
        for freq in self._doc_freqs:
            weights = {token: (1 + math.log(tf)) * self._idf(token) for token, tf in freq.items()}
            norm = math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
            self._doc_weights.append({token: weight / norm for token, weight in weights.items()})

    def _idf(self, token):
        df = self._df.get(token, 0)
        return math.log((1 + self._doc_count) / (1 + df)) + 1.0

    def search(self, query, top_k=5):
        query = (query or "").strip()
        if not query or not self._doc_count:
            return []
        freq = {}
        for token in tokenize(query):
            freq[token] = freq.get(token, 0) + 1
        if not freq:
            return []
        weights = {token: (1 + math.log(tf)) * self._idf(token) for token, tf in freq.items()}
        norm = math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
        query_weights = {token: weight / norm for token, weight in weights.items()}

        scores = []
        for index, doc_weights in enumerate(self._doc_weights):
            score = sum(weight * doc_weights[token] for token, weight in query_weights.items() if token in doc_weights)
            if score > 0:
                scores.append((score, index))

        # 排序规则与 BM25 一致：分数降序，同分按 id 稳定可复现
        scores.sort(key=lambda item: (-item[0], self._entries[item[1]]["id"]))
        top_k = max(1, min(int(top_k), 20))
        hits = []
        for score, index in scores[:top_k]:
            entry = self._entries[index]
            hits.append(
                {
                    "id": entry["id"],
                    "title": entry["title"],
                    "category": entry.get("category", ""),
                    "score": round(score, 3),
                    "source": entry.get("source_file", ""),
                    "content": entry.get("content", ""),
                }
            )
        return hits
