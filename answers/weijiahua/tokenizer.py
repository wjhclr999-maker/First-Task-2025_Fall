import json
from typing import List, Dict, Tuple, Iterable, Iterator
import regex as re
    
# ==========================================
# 1. 工具函数：BPE 合并相关
# ==========================================

def best_merge_pair(seq, pair_rank):
    """
    在当前字节序列 seq 中，找到优先级最高的一对相邻 bytes (A, B)。
    pair_rank 是一个字典：{(A, B): rank}，rank 越小优先级越高。
    找不到任何可以合并的字节对时返回 None。
    """
    best = None
    best_rank = None
    for i in range(len(seq) - 1):
        p = (seq[i], seq[i + 1])
        if p in pair_rank:
            r = pair_rank[p]
            if best is None or r < best_rank:
                best = p
                best_rank = r
    return best


def merge_once_sequence(seq, A, B):
    """
    对一个字节序列 seq 应用一次合并规则 (A, B)：
    遍历 seq 中的相邻元素，遇到 A, B 连在一起就把它们合并成 A+B。
    返回合并后的新序列（tuple）。
    """
    AB = A + B
    out = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == A and seq[i + 1] == B:
            out.append(AB)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def bpe_encode_bytes(seq, merges):
    """
    训练阶段用到的 BPE 编码函数：
    给定一串 bytes（被拆成单字节的 tuple），以及 merges 列表，
    按 merges 的顺序不断合并，直到没有可合并的字节对。
    """
    pair_rank = {pair: rank for rank, pair in enumerate(merges)}

    while True:
        p = best_merge_pair(seq, pair_rank)
        if p is None:
            break
        seq = merge_once_sequence(seq, p[0], p[1])
    return seq


# ==========================================
# 2. Tokenizer 类：编码 & 解码
# ==========================================

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        当你运行 Tokenizer(...) 时，这个函数会被调用。
        它负责把你训练好的 词表 和 合并规则 存到这个对象里。
        """
        self.vocab = vocab
        self.merges = merges

        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        
        # bytes -> id（后面编码时会用）
        self.vocab_inverse = {v: k for k, v in vocab.items()}

        # merges 中每一对 (A, B) 的优先级（索引越小，rank 越小）
        self.pair_rank = {pair: rank for rank, pair in enumerate(self.merges)}

        # === 训练时的预分词正则：必须和 train_bpe 中使用的一致 ===
        # 这里使用 GPT-2 的正则表达式。
        self._pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| """
            r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    # ---------- 内部小工具：BPE & 预分词 ----------

    def _bpe_bytes(self, seq: Tuple[bytes, ...]) -> Tuple[bytes, ...]:
        """
        在当前 merges / pair_rank 下，对一个字节序列执行完整的 BPE 合并。
        """
        while True:
            p = best_merge_pair(seq, self.pair_rank)
            if p is None:
                break
            A, B = p
            seq = merge_once_sequence(seq, A, B)
        return seq

    def _encode_without_specials(self, text: str) -> List[int]:
        """
        对一段不包含任何 special token 的普通文本进行编码：
        1. 用训练时的正则 self._pat 预分词
        2. 每个分词块转成 UTF-8 bytes
        3. bytes 上跑 BPE
        4. 查 vocab_inverse 得到 ID
        """
        ids: List[int] = []
        for token in self._pat.findall(text):
            if not token:
                continue
            b = token.encode("utf-8")
            seq = tuple(bytes([byte_val]) for byte_val in b)
            merged_seq = self._bpe_bytes(seq)
            for token_bytes in merged_seq:
                try:
                    token_id = self.vocab_inverse[token_bytes]
                except KeyError:
                    raise KeyError(f"Unknown token bytes in vocab: {token_bytes!r}")
                ids.append(token_id)
        return ids

    def _split_by_special_tokens(self, text: str):
        """
        把文本拆成若干块：
        [(is_special: bool, segment: str), ...]
        True 表示这一段是一个完整的 special token 字符串（例如 "<|endoftext|>"）。
        对重叠的 special tokens 采用“最长匹配优先”的策略。
        """
        if not self.special_tokens:
            return [(False, text)]

        result = []
        i = 0
        start = 0
        n = len(text)

        while i < n:
            matched = None
            # 关键：按长度从长到短匹配，重叠时优先选最长的那个
            for tok in sorted(self.special_tokens, key=len, reverse=True):
                if text.startswith(tok, i):
                    matched = tok
                    break

            if matched is not None:
                # 先把 special token 之前的普通文本收集起来
                if i > start:
                    result.append((False, text[start:i]))
                # 再把 special token 本身作为一个整体收集
                result.append((True, matched))
                i += len(matched)
                start = i
            else:
                i += 1

        # 收尾剩余的普通文本
        if start < n:
            result.append((False, text[start:]))

        return result


    # ---------- 对外接口：encode / decode / encode_iterable ----------

    def encode(self, text: str) -> List[int]:
        """
        编码文本（Encoding text）：
        1. 使用训练时相同的正则 self._pat 预分词
        2. 对每个预分词块应用 merges（BPE）
        3. 同时正确处理 special tokens：它们作为整体映射到单个 ID
        """
        ids: List[int] = []

        for is_special, segment in self._split_by_special_tokens(text):
            if not segment:
                continue

            if is_special:
                # special token 直接用 bytes 查 vocab
                token_bytes = segment.encode("utf-8")
                try:
                    token_id = self.vocab_inverse[token_bytes]
                except KeyError:
                    raise KeyError(f"Special token {segment!r} not found in vocab")
                ids.append(token_id)
            else:
                ids.extend(self._encode_without_specials(segment))

        return ids

    def decode(self, ids) -> str:
        """
        解码文本（Decoding text）：
        支持两种输入形式：
        - 一维： [1, 2, 3, ...]
        - 嵌套： [[1, 2], [3, 4], ...]  （会自动拍扁）
        1. 遍历所有 token ID
        2. 用 vocab 查出对应的 bytes
        3. 拼接成一个大的 bytes 串
        4. 使用 .decode("utf-8", errors="replace") 得到最终字符串
        """

        def _flatten(seq):
            for x in seq:
                if isinstance(x, (list, tuple)):
                    # 递归拍扁嵌套 list/tuple
                    yield from _flatten(x)
                else:
                    yield x

        chunks = []
        for token_id in _flatten(ids):
            try:
                chunks.append(self.vocab[token_id])
            except KeyError:
                raise KeyError(f"Unknown token id in vocab: {token_id}")
        data = b"".join(chunks)
        return data.decode("utf-8", errors="replace")


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对一个字符串可迭代对象进行流式编码。
        返回一个“按顺序产出 token id 的迭代器”，而不是“id 列表的列表”。
        这样：
            list(tokenizer.encode_iterable(...))  -> [id0, id1, id2, ...]
        既满足 matches_tiktoken，又能通过内存使用测试。
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id


    # ---------- 从文件加载 vocab / merges ----------

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        从文件加载 Tokenizer。
        假设：
        - vocab.json: {"0": [byte_values], ...} 或 {"0": "<latin1_str>", ...}
        - merges.json: 可能是
            1) [[ [b1,...], [c1,...] ], ...]           # 直接存的字节列表
            2) [[ "latin1_str1", "latin1_str2" ], ...] # 存成 latin-1 字符串
            3) [[ id1, id2 ], ...]                     # 存的是 vocab 里的 token id
        """
        # 1. 读取词表
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_raw = json.load(f)

        vocab: Dict[int, bytes] = {}
        for id_str, byte_list in vocab_raw.items():
            # JSON 里无法直接存 bytes，一般会存成“整数列表”或“latin-1 字符串”
            if isinstance(byte_list, list):
                vocab[int(id_str)] = bytes(byte_list)
            else:
                vocab[int(id_str)] = byte_list.encode("latin-1")

        # 2. 读取合并规则
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_raw = json.load(f)

        merges: List[Tuple[bytes, bytes]] = []
        for pair in merges_raw:
            a, b = pair
            if isinstance(a, list) and isinstance(b, list):
                # 情况 1: 直接存的字节列表
                A = bytes(a)
                B = bytes(b)
            elif isinstance(a, str) and isinstance(b, str):
                # 情况 2: 存成 latin-1 字符串
                A = a.encode("latin-1")
                B = b.encode("latin-1")
            elif isinstance(a, int) and isinstance(b, int):
                # 情况 3: 存的是 vocab 的 id
                A = vocab[a]
                B = vocab[b]
            else:
                raise TypeError(f"Unexpected merge pair format: {pair!r}")
            merges.append((A, B))

        # 3. 创建 Tokenizer 实例
        return cls(vocab, merges, special_tokens)
