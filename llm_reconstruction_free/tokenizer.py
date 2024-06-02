from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex,
)
import transformers
from tqdm import tqdm

from . import gcs

SPECIAL_TOKENS = dict(
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    bos_token="<s>",
    eos_token="</s>",
)
NAMES = ["wordpiece", "identity", "BPE"]


def get_normalizer(lower_case=True, accents=True, quotes=True):
    sequence = []
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


def wrap(
    tokenizer,
):
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        **SPECIAL_TOKENS,
        padding_side="right",
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return tok


class SplitAll:

    def split(self, i, s):
        splits = []
        for l in range(len(str(s))):
            splits.append(s[l : l + 1])
        return splits

    def pre_tokenize(self, pretok):
        pretok.split(self.split)


class JoinAll:
    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode_chain(self, tokens):
        return tokens


def train_identity(training_corpus, **kwargs):
    for name, value in kwargs.items():
        print(f" argument {name} with value {value} is ignored")

    normalizer = get_normalizer()
    uni = set()
    for i in tqdm(training_corpus):
        if type(i) != list:
            i = [i]
        for sub in i:
            uni.update(list(normalizer.normalize_str(sub)))
    vocab = dict()
    for i, token in enumerate(uni):
        vocab[token] = i
    vocab[SPECIAL_TOKENS["unk_token"]] = i + 1
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab))

    tokenizer.normalizer = normalizer

    encoding = tokenizer.encode("Let's test this tokenizer.")

    # workaround based on https://github.com/huggingface/tokenizers/issues/581
    tokenizer = wrap(tokenizer)
    tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(SplitAll())
    tokenizer._tokenizer.decoder = decoders.Decoder.custom(JoinAll())
    return tokenizer


def train_wordpiece(training_corpus, vocab_size):
    tokenizer = Tokenizer(models.WordPiece(unk_token=SPECIAL_TOKENS["unk_token"]))
    tokenizer.normalizer = get_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=list(SPECIAL_TOKENS.values())
    )
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

    print(encoding.tokens)
    print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test this pre-tokenizer."))
    print(tokenizer.decode(encoding.ids))
    #    assert tokenizer.decode(encoding.ids) == "Let's test this tokenizer."

    return wrap(tokenizer)


def train_BPE(training_corpus, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = get_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[SEP]"])
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)
    print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test this pre-tokenizer."))
    print(tokenizer.decode(encoding.ids))
    # assert tokenizer.decode(encoding.ids) == "Let's test this tokenizer."

    return wrap(tokenizer)
    # tokenizer.save("tokenizer.json")
    # new_tokenizer = Tokenizer.from_file("tokenizer.json")


def from_model(name, from_gcs: str = None):
    if "apple" in name:
        print("For apple model we override to the llama-2 one")
        name = "meta-llama/Llama-2-7b-hf"
    if from_gcs:
        local_cache = gcs.local_copy(from_gcs, "tokenizers", name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                local_cache, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                name, trust_remote_code=True)

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def from_data(dataset, variant: str, vocab_size: int):

    assert variant in NAMES
    assert "text" in dataset.column_names

    def dataset_iterator():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    if variant == "wordpiece":
        return train_wordpiece(dataset_iterator(), vocab_size=vocab_size)
    elif variant == "identity":
        return train_identity(dataset_iterator(), vocab_size=vocab_size)
    else:
        return train_BPE(dataset_iterator(), vocab_size=vocab_size)
