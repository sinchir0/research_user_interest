import json
from typing import Set
from urllib.request import urlopen

import gensim
import luigi
import numpy as np
from gokart.config_params import inherits_config_params
from research_user_interest.model.config import MasterConfig
from research_user_interest.model.get_pretrain_vector import ExtractPretrainedVectorTask
from research_user_interest.model.preprocess import CleanBodyTask
from research_user_interest.utils.template import GokartTask
from sudachipy import dictionary, tokenizer

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C


class AddNoumColTask(GokartTask):

    slothlib_path: str = luigi.Parameter()
    append_word: list = luigi.Parameter()

    def requires(self):
        return CleanBodyTask()

    def parse_text(self, text: str) -> list:
        return [
            m.surface()
            for m in tokenizer_obj.tokenize(text, mode)
            if m.part_of_speech()[0] == "名詞"
        ]

    def delete_dup_word(self, word_list: list) -> list:
        return list(dict.fromkeys(word_list))

    def remove_stopword(self, word_list: list, stopword_list: list) -> list:
        return [word for word in word_list if word not in stopword_list]

    def run(self):
        df = self.load()

        # 名詞のみ抽出する
        df["clean_body_parse_noum"] = df["clean_body"].apply(
            lambda x: self.parse_text(x)
        )

        # 重複した単語を削除する
        df["clean_body_parse_noum"] = df["clean_body_parse_noum"].apply(
            lambda x: self.delete_dup_word(x)
        )

        # stopwordを削除する
        slothlib_file = urlopen(self.slothlib_path)
        stopword_list = [
            line.decode("utf-8").strip()
            for line in slothlib_file
            if not line.decode("utf-8").strip() == u""
        ]
        stopword_list.extend(json.loads(self.append_word))

        df["clean_body_parse_noum"] = df["clean_body_parse_noum"].apply(
            lambda x: self.remove_stopword(x, stopword_list)
        )

        self.dump(df)


class AddNamedEntityColTask(GokartTask):
    def requires(self):
        return AddNoumColTask()

    def parse_text(self, text: str) -> list:
        return [
            m.surface()
            for m in tokenizer_obj.tokenize(text, mode)
            if m.part_of_speech()[1] == "固有名詞"
        ]

    def delete_dup_word(self, word_list: list) -> list:
        return list(dict.fromkeys(word_list))

    def run(self):
        df = self.load()

        # 名詞のみ抽出する
        df["clean_body_parse_named_entity"] = df["clean_body"].apply(
            lambda x: self.parse_text(x)
        )

        # 重複した単語を削除する
        df["clean_body_parse_named_entity"] = df["clean_body_parse_named_entity"].apply(
            lambda x: self.delete_dup_word(x)
        )

        self.dump(df)


@inherits_config_params(MasterConfig)
class AddVectorTask(GokartTask):
    model_path: str = luigi.Parameter()
    embedding_dim: int = luigi.IntParameter()

    oov_initialize_range: set = (-0.01, 0.01)

    def requires(self):
        return {
            "dataframe": AddNamedEntityColTask(),
            "_pretraind_vector": ExtractPretrainedVectorTask(),
        }

    def get_word_mean_embedding(
        self,
        model,
        vocab: Set[str],
        embedding_dim: int,
        oov_initialize_range: Set[int],
        text_list: list,
    ):

        vectors = []
        for word in text_list:
            if word in vocab:
                vectors.append(model[word])
            else:
                vectors.append(
                    np.random.uniform(
                        oov_initialize_range[0],
                        oov_initialize_range[1],
                        embedding_dim,
                    )
                )

        vector_array = np.array(vectors)

        # 平均を取得
        vector_mean = np.mean(vector_array, axis=0)

        return vector_mean

    def run(self):
        np.random.seed(0)
        df = self.load("dataframe")

        model = gensim.models.KeyedVectors.load_word2vec_format(
            self.model_path, binary=True
        )

        vocab = set(model.index_to_key)

        df["noum_vector_mean"] = df["clean_body_parse_noum"].apply(
            lambda x: self.get_word_mean_embedding(
                model=model,
                vocab=vocab,
                embedding_dim=self.embedding_dim,
                oov_initialize_range=self.oov_initialize_range,
                text_list=x,
            )
        )

        df["named_entity_vector_mean"] = df["clean_body_parse_named_entity"].apply(
            lambda x: self.get_word_mean_embedding(
                model=model,
                vocab=vocab,
                embedding_dim=self.embedding_dim,
                oov_initialize_range=self.oov_initialize_range,
                text_list=x,
            )
        )

        self.dump(df)
