#!/usr/bin/env python
# coding: utf-8

def select_blocking_tokens(refDict, tokenFreqDict, minBlkTokenLen=4,
                           excludeNumericBlocks=True, beta=2):
    """
    Collect the set of all unique blocking tokens across every record.

    A token qualifies as a blocking token when:
      - its length >= minBlkTokenLen
      - it is not all-digits (when excludeNumericBlocks is True)
      - its corpus frequency is in the range [2, beta]

    Returns a sorted list of unique qualifying tokens that forms the
    one-hot encoding vocabulary.
    """
    vocab = set()
    for refID, tokenList in refDict.items():
        for token in tokenList:
            if len(token) < minBlkTokenLen:
                continue
            if excludeNumericBlocks and token.isdigit():
                continue
            freq = tokenFreqDict.get(token, 0)
            if freq < 2 or freq > beta:
                continue
            vocab.add(token)
    return sorted(vocab)


def build_one_hot_vectors(refDict, tokenFreqDict, minBlkTokenLen=4,
                          excludeNumericBlocks=True, beta=2):
    """
    Build a one-hot encoding vector for every record in refDict.

    The vocabulary is the sorted set of all blocking tokens found across
    all records (see select_blocking_tokens).  Each position in the vector
    corresponds to one vocabulary token.  A position is set to 1 when that
    blocking token is present in the record, 0 otherwise.

    Example
    -------
    vocab   = ['john', 'smith']
    record  = ['John', 'Smith', 'Atlanta', 'GA']   (tokens already lowered upstream)
    vector  = [1, 1]

    Parameters
    ----------
    refDict            : dict  {refID: [token, ...]}
    tokenFreqDict      : dict  {token: frequency}
    minBlkTokenLen     : int   minimum token length to qualify (default 4)
    excludeNumericBlocks: bool  exclude all-digit tokens (default True)
    beta               : int   maximum frequency to qualify (default 2)

    Returns
    -------
    vocab   : list[str]          sorted vocabulary of blocking tokens
    vectors : dict[str, list[int]]
                {refID: one-hot vector aligned to vocab}
    """
    vocab = select_blocking_tokens(refDict, tokenFreqDict,
                                   minBlkTokenLen, excludeNumericBlocks, beta)

    if not vocab:
        return vocab, {refID: [] for refID in refDict}

    # Map each blocking token to its index in the vector
    token_index = {token: idx for idx, token in enumerate(vocab)}
    vocab_size = len(vocab)

    vectors = {}
    for refID, tokenList in refDict.items():
        vec = [0] * vocab_size
        for token in tokenList:
            if token in token_index:
                vec[token_index[token]] = 1
        vectors[refID] = vec

    return vocab, vectors
