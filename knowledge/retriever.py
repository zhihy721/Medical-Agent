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
                }
            )
        return hits


# 默认全局检索器：模块加载时建索引，同 query 结果确定可复现
DEFAULT_RETRIEVER = BM25Retriever()
