import torch


def run(self, words, embedding_file):

    words = {w for w in words if w in self.word_dict}
    print('Loading pre-trained embeddings for %d words from %s' % (len(words), embedding_file))
    embedding = self.network.embedding.weight.data

    # When normalized, some words are duplicated. (Average the embeddings).
    vec_counts = {}

    with open(embedding_file, encoding="utf8") as f:
        # Skip first line if of form count/dim.
        line = f.readline().rstrip().split(' ')
        if len(line) != 2:
            f.seek(0)

        for line in f:
            parsed = line.rstrip().split(' ')
            assert(len(parsed) == embedding.size(1) + 1)
            w = self.word_dict.normalize(parsed[0])
            if w in words:
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                if w not in vec_counts:
                    vec_counts[w] = 1
                    embedding[self.word_dict[w]].copy_(vec)
                else:
                    logging.warning(
                        'WARN: Duplicate embedding found for %s' % w
                    )
                    vec_counts[w] = vec_counts[w] + 1
                    embedding[self.word_dict[w]].add_(vec)

    for w, c in vec_counts.items():
        embedding[self.word_dict[w]].div_(c)
    torch.save(embedding, open("embedding.mdl", "wb"))