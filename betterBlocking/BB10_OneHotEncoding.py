#!/usr/bin/env python
# coding: utf-8

import scipy.sparse


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
    Build a sparse one-hot matrix for every record in refDict.

    Each blocking token in a record gets its own one-hot row vector —
    a vector of length len(vocab) with exactly one 1 at the token's
    vocabulary index and 0s elsewhere.  A record with k blocking tokens
    produces a matrix of shape (k, vocab_size).

    Records with no blocking tokens get an empty matrix of shape (0, vocab_size).

    All matrices are stored as scipy CSR sparse matrices.

    Example
    -------
    vocab  = ['john', 'smith']          # indices 0, 1
    record = ['john', 'smith', 'atlanta', 'ga']

    john  → sparse row [1, 0]
    smith → sparse row [0, 1]
    matrix shape = (2, 2)

    Parameters
    ----------
    refDict              : dict  {refID: [token, ...]}
    tokenFreqDict        : dict  {token: frequency}
    minBlkTokenLen       : int   minimum token length to qualify (default 4)
    excludeNumericBlocks : bool  exclude all-digit tokens (default True)
    beta                 : int   maximum frequency to qualify (default 2)

    Returns
    -------
    vocab   : list[str]
                sorted vocabulary of blocking tokens
    vectors : dict[str, scipy.sparse.csr_matrix]
                {refID: sparse matrix of shape (k, vocab_size)}
                each row is the one-hot vector for one blocking token
    """
    vocab = select_blocking_tokens(refDict, tokenFreqDict,
                                   minBlkTokenLen, excludeNumericBlocks, beta)
    vocab_size = len(vocab)
    token_index = {token: idx for idx, token in enumerate(vocab)}

    vectors = {}
    for refID, tokenList in refDict.items():
        blocking = [t for t in tokenList if t in token_index]
        if not blocking:
            vectors[refID] = scipy.sparse.csr_matrix((0, vocab_size))
        else:
            rows = list(range(len(blocking)))
            cols = [token_index[t] for t in blocking]
            data = [1] * len(blocking)
            vectors[refID] = scipy.sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(blocking), vocab_size),
                dtype=int
            )

    return vocab, vectors


if __name__ == '__main__':
    import argparse, re

    parser = argparse.ArgumentParser(description='One-hot encode records by blocking tokens')
    parser.add_argument('inputFile',           help='CSV input file (refID in first column)')
    parser.add_argument('--delimiter', default=',',  help='Field delimiter (default: ,)')
    parser.add_argument('--hasHeader', action='store_true', help='Skip first line as header')
    parser.add_argument('--beta',      type=int, default=2, help='Max token frequency (default: 2)')
    parser.add_argument('--minLen',    type=int, default=4, help='Min token length (default: 4)')
    parser.add_argument('--allowNums', action='store_true', help='Allow all-digit tokens')
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
    for refID, mat in vectors.items():
        if mat.shape[0] == 0:
            print(f"{refID}  blocking_tokens=[]  (no qualifying tokens)")
        else:
            for row_idx in range(mat.shape[0]):
                row = mat.getrow(row_idx)
                token = vocab[row.indices[0]]
                print(f"{refID}  token='{token}'  one-hot={row.toarray()[0].tolist()}")
