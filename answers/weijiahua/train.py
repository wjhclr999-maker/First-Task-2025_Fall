import re
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[List[Tuple[bytes, bytes]], Dict[int, bytes]]:
    r"""
    Problem 3: BPE 分词器训练

    - 初始词表：0~255 共 256 个字节，每个字节一个 token。
    - 预分词：
        1) 如果有 special_tokens，先用 re.split 按这些特殊 token 把整篇文本拆块；
        2) 对非特殊块再用 GPT-2 风格正则进行预分词；
        3) BPE 的合并只在每个预分词块（word/子串）内部进行，不会跨越边界。
    - Special tokens：例如 <|endoftext|>，
        * 作为独立 token 加入 vocab；
        * 在语料里先被 re.split 单独切出来；
        * 在 BPE 训练中始终是长度为 1 的 ID 序列，不会被拆开，也不会和别的 token 合并。
    - 合并规则（BPE merges）：
        每一轮：
          1) 统计所有 token 序列中相邻 token 对的频率；
          2) 找到频率最高的 pair；
             如果有平局，选择字典序更大的那一对（按 bytes 比较）；
          3) 为该 pair 分配一个新的 token ID，并在所有序列中用新 ID 替换该 pair；
          4) 把这个 pair 对应的 (bytes, bytes) 记录到 merges 列表。
        重复直到达到 vocab_size 或没有可合并的 pair。
    """

    # ===== 1. 读取文本 =====
    path_obj = Path(input_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"找不到文件: {input_path}")
    text = path_obj.read_text(encoding="utf-8")

    # ===== 2. GPT-2 风格预分词正则（Python 版本近似 \p{L}, \p{N}）=====
    # 原始题目给的是：
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # 这里因为 Python 不直接支持 \p{L}/\p{N}，用 \w / \d 近似，对英文语料足够。
    gpt2_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    gpt2_re = re.compile(gpt2_pat)

    # ===== 3. 先按 special_tokens 用 re.split 拆分语料，再对普通块做预分词 =====
    words: List[str] = []

    if special_tokens:
        # 构造形如 '(<|endoftext|>)' 的捕获组，这样 split 之后 special token 本身也会保留在结果列表里
        special_pat = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        split_re = re.compile(special_pat)

        for chunk in split_re.split(text):
            if not chunk:
                continue
            if chunk in special_tokens:
                # 遇到特殊 token，直接作为一个“预 token”
                words.append(chunk)
            else:
                # 普通文本块，用 GPT-2 regex 进一步预分词
                words.extend(gpt2_re.findall(chunk))
    else:
        # 没有特殊 token 时，就直接对整篇文本做 GPT-2 预分词
        words = gpt2_re.findall(text)

    # ===== 4. 初始化 vocab：256 字节 + special tokens =====
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    special_token_ids: Dict[str, int] = {}
    next_id = 256

    for tok in special_tokens:
        tok_bytes = tok.encode("utf-8")
        vocab[next_id] = tok_bytes
        special_token_ids[tok] = next_id
        next_id += 1

    # ===== 5. 把每个“预 token”映射为 ID 序列，并统计频率 =====
    # seqs_freq: dict[tuple[int, ...], int] —— 每种 ID 序列出现了多少次
    word_counts = Counter(words)
    seqs_freq: Dict[Tuple[int, ...], int] = {}

    for word, freq in word_counts.items():
        if word in special_token_ids:
            # special token：整个 token 用一个 ID 表示，长度为 1
            seq = (special_token_ids[word],)
        else:
            # 普通 token：编码为 UTF-8 字节，再把每个字节映射到 0~255 的原始 vocab ID
            b = word.encode("utf-8")
            seq = tuple(b[i] for i in range(len(b)))
        seqs_freq[seq] = freq

    merges: List[Tuple[bytes, bytes]] = []

    # 还可以增加的 token 数 = 目标 vocab 大小 - 当前 vocab 大小
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        # 目标 vocab_size 比当前还小，就不做合并了
        return merges, vocab

    # ===== 6. BPE 合并主循环 =====
    for _ in range(num_merges):
        # 6.1 统计所有序列中“相邻 token 对”的频次
        pairs: Counter[Tuple[int, int]] = Counter()
        for seq, freq in seqs_freq.items():
            if len(seq) < 2:
                continue
            for j in range(len(seq) - 1):
                pair = (seq[j], seq[j + 1])
                pairs[pair] += freq

        # 没有任何可以合并的 pair 了
        if not pairs:
            break

        # 6.2 选出“频率最高 + 字典序更大”的 pair
        # 作业要求：平局时取 lexicographically greater pair
        # 我们按 (freq, (bytes1, bytes2)) 比较：
        #   - 先比频率；
        #   - 再按两个 bytes 串的字典序比较。
        best_pair = max(
            pairs.keys(),
            key=lambda p: (pairs[p], (vocab[p[0]], vocab[p[1]])),
        )
        id1, id2 = best_pair

        # 6.3 为 best_pair 分配一个新 token ID，并记录 merge 规则
        new_id = next_id
        next_id += 1

        # 新 token 的 bytes = vocab[id1] + vocab[id2]
        vocab[new_id] = vocab[id1] + vocab[id2]
        merges.append((vocab[id1], vocab[id2]))

        # 6.4 在所有序列中，用 new_id 替换 (id1, id2)
        new_seqs_freq: Dict[Tuple[int, ...], int] = {}

        for seq, freq in seqs_freq.items():
            # 小优化：如果 seq 里根本没有 id1，就不可能出现 (id1, id2)
            if id1 not in seq:
                new_seqs_freq[seq] = freq
                continue

            new_seq: List[int] = []
            k = 0
            L = len(seq)

            while k < L:
                if k < L - 1 and seq[k] == id1 and seq[k + 1] == id2:
                    # 发现一个 (id1, id2)，合并成 new_id
                    new_seq.append(new_id)
                    k += 2
                else:
                    new_seq.append(seq[k])
                    k += 1

            new_seqs_freq[tuple(new_seq)] = freq

        seqs_freq = new_seqs_freq

    return merges, vocab
