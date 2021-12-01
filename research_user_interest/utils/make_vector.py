from enum import Enum
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from swem import SWEM, MeCabTokenizer

class Method(Enum):
    AVERAGE = "average"
    MAX = "max"

class MakeWordVector():
    def __init__(self):
        self.w2v_path = Path(__file__).resolve().parents[1].joinpath("user_vec", "models", "bin", "entity_vector.model.bin")
        self.tokenizer = MeCabTokenizer("-O wakati")
        self.w2v_model = KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)
        self.swem = SWEM(self.w2v_model, self.tokenizer)

    def make_vector(self, model, input_text: str, method: str):
        """入力されたmodelを用いてtextをvectorに変換する"""
        if method == Method.AVERAGE:
            return model.average_pooling(input_text)
        elif method == Method.MAX:
            return model.max_pooling(input_text)
        else:
            raise NotImplementedError

    def make_text_vector(self, text_list: str, method: Method = Method.AVERAGE):
        """text_listをvectorを付与する"""
        return make_vector_by_swem(model=swem, input_text=text_list, method=method)

    def calc_vector_mean(self, vector):
        """入力されたvectorのmeanを計算する"""
        vector = np.stack(vector)
        vector_mean = np.average(vector, axis=0)
        return vector_mean

    # def make_word_vector(df, method: Method = Method.AVERAGE):
    #     """dfにvectorを付与する"""
    #     # swemモデルの構築
    #     w2v_path = Path(__file__).resolve().parents[1].joinpath("user_vec", "models", "bin", "entity_vector.model.bin")
    #     tokenizer = MeCabTokenizer("-O wakati")
    #     w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    #     swem = SWEM(w2v_model, tokenizer)

    #     # text毎にvectorを付与
    #     df["text_vec"] = df["text"].apply(
    #         lambda text: make_vector_by_swem(model=swem, input_text=text, method=method)
    #     )
    #     return df