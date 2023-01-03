from typing import List


class TextEncoder:
    """
    テキストをエンコードするクラス。
    """

    def __init__(self):
        self._vocab_list = [
            "<pad>",
            "<sos>",
            "<eos>",
            " ",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "'",
        ]
        self._vocab_to_index = {v: index for index, v in enumerate(self._vocab_list)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_list)

    @property
    def pad_index(self) -> int:
        return 0

    @property
    def sos_index(self) -> int:
        return 1

    @property
    def eos_index(self) -> int:
        return 2

    def vocab_to_index(self, vocab: str) -> int:
        """
        Arguments:
            vocab: str
                語彙の文字
        Returns:
            index: int
                語彙のインデックス
        """
        return self._vocab_to_index.get(vocab, self.pad_index)

    def index_to_vocab(self, index: int) -> str:
        """
        Arguments:
            index: int
                語彙のインデックス
        Returns:
            vocab: str
                語彙の文字
        """
        return self._vocab_list[index]

    def encode(self, s: str) -> List[int]:
        """
        Arguments:
            s: str
                エンコードするテキスト
        Returns:
            indexes: List[int]
                エンコードされたテキスト
        """
        s = s.strip("\r\n ")
        return [self.vocab_to_index(v) for v in s] + [self.eos_index]

    def decode(self, indexes: List[int], ignore_repeat: bool = False) -> str:
        """
        Arguments:
            indexes: List[int]
                デコードするテキスト
            ignore_repeat: bool
                繰り返しを無視するかどうか
        Returns:
            s: str
                デコードされたテキスト
        """
        vocabs = []
        for t, index in enumerate(indexes):
            v = self.index_to_vocab(index)
            if index == self.pad_index or (
                ignore_repeat and t > 0 and index == indexes[t - 1]
            ):
                continue
            elif index == self.eos_index:
                break
            else:
                vocabs.append(v)
        return "".join(vocabs)
