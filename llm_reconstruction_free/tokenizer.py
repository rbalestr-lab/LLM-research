from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


def get_normalizer(lower_case=False, accents=True, quotes=True):
    seqeunce = []
    if quotes:
        sequence.append(normalizers.Replace("``", '"'))
        sequence.append(normalizers.Replace("''", '"'))
    sequence.append(normalizers.NFKD())
    sequence.append(normalizers.Replace(Regex(" {2,}"), " "))
    if lower_case:
        sequence.append(normalizers.Lowercase())
    if accents:
        sequence.append(normalizers.StripAccents())
    return normalizers.Sequence(sequence)


def train_wordpiece(vocab_size):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(get_normalizer())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    encoding = tokenizer.encode("Let's test this tokenizer.")
    assert tokenizer.decode(encoding.ids) == "Let's test this tokenizer."

    return tokenizer


def train_BPE():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence(get_normalizer())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[SEP]"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    encoding = tokenizer.encode("Let's test this tokenizer.")
    assert tokenizer.decode(encoding.ids) == "Let's test this tokenizer."

    return tokenizer
    #tokenizer.save("tokenizer.json")
    #new_tokenizer = Tokenizer.from_file("tokenizer.json")
