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


if __name__ == '__main__':
    import argparse, re, sys

    parser = argparse.ArgumentParser(description='One-hot encode records by blocking tokens')
    parser.add_argument('inputFile',            help='CSV input file (refID in first column)')
    parser.add_argument('--delimiter',  default=',', help='Field delimiter (default: ,)')
    parser.add_argument('--hasHeader',  action='store_true', help='Skip first line as header')
    parser.add_argument('--beta',       type=int,   default=2, help='Max token frequency (default: 2)')
    parser.add_argument('--minLen',     type=int,   default=4, help='Min token length (default: 4)')
    parser.add_argument('--allowNums',  action='store_true',   help='Allow all-digit tokens')
    args = parser.parse_args()

    # Tokenize input
    refDict = {}
    with open(args.inputFile) as f:
        if args.hasHeader:
            f.readline()
        for line in f:
            parts = line.strip().split(args.delimiter)
            if not parts or not parts[0]:
                continue
            refID = parts[0]
            body = ' '.join(parts[1:])
            tokens = re.sub(r'\W', ' ', body.lower()).split()
            refDict[refID] = tokens

    # Build token frequency dictionary
    tokenFreqDict = {}
    for tokens in refDict.values():
        for t in tokens:
            tokenFreqDict[t] = tokenFreqDict.get(t, 0) + 1

    vocab, vectors = build_one_hot_vectors(
        refDict, tokenFreqDict,
        minBlkTokenLen=args.minLen,
        excludeNumericBlocks=not args.allowNums,
        beta=args.beta
    )

    print(f"Vocabulary ({len(vocab)} tokens): {vocab}\n")
    for refID, vec in vectors.items():
        active = [vocab[i] for i, v in enumerate(vec) if v == 1]
        print(f"{refID}  blocking_tokens={active}  vector={vec}")
