from logging import getLogger

import luigi
import numpy as np
from annoy import AnnoyIndex
from gokart.config_params import inherits_config_params
from research_user_interest.model.config import MasterConfig
from research_user_interest.model.extract_text_info import AddVectorTask
from research_user_interest.utils.template import GokartTask

logger = getLogger(__name__)


@inherits_config_params(MasterConfig)
class GetSimilarArticleByAnnoyTask(GokartTask):
    embedding_dim: int = luigi.IntParameter()
    target_title1: str = luigi.Parameter()
    target_title2: str = luigi.Parameter()
    target_title1_label: str = luigi.Parameter()
    target_title2_label: str = luigi.Parameter()

    metric: str = luigi.Parameter()
    article_num: int = luigi.IntParameter()

    def requires(self):
        return AddVectorTask()

    def do_annoy(self, df, col: str, target_title: str, label: str = ""):
        if label == "":
            logger.info(f"do annoy for {target_title[:10]} by {col}")
        else:
            logger.info(f"do annoy for {target_title[:10]} label:{label} by {col}")
            df = df[df["label"] == label].reset_index(drop=True)

        logger.info("")

        annoy_index = AnnoyIndex(self.embedding_dim, metric=self.metric)

        target_index = df[df["title"].str.contains(target_title)].index.values[0]

        for idx, vector in enumerate(df[col]):

            # targetのindexは学習しない
            if idx == target_index:
                continue

            # vectorがnanの場合を除く
            if type(vector) != np.ndarray:
                if np.isnan(vector):
                    continue

            annoy_index.add_item(idx, vector.tolist())

        annoy_index.build(n_trees=10)

        simlar_idx_list = annoy_index.get_nns_by_vector(
            df.loc[target_index, col].tolist(), self.article_num
        )

        for idx in simlar_idx_list:
            logger.info(df.loc[idx, "title"])
        logger.info("----------")

    def run(self):
        df = self.load()

        # indexが順番通りになるよう修正
        df = df.reset_index(drop=True)

        # title1
        self.do_annoy(df, col="noum_vector_mean", target_title=self.target_title1)

        self.do_annoy(
            df, col="named_entity_vector_mean", target_title=self.target_title1
        )

        self.do_annoy(
            df,
            col="noum_vector_mean",
            target_title=self.target_title1,
            label=self.target_title1_label,
        )

        self.do_annoy(
            df,
            col="named_entity_vector_mean",
            target_title=self.target_title1,
            label=self.target_title1_label,
        )

        # title2
        self.do_annoy(df, col="noum_vector_mean", target_title=self.target_title2)

        self.do_annoy(
            df, col="named_entity_vector_mean", target_title=self.target_title2
        )

        self.do_annoy(
            df,
            col="noum_vector_mean",
            target_title=self.target_title2,
            label=self.target_title2_label,
        )

        self.do_annoy(
            df,
            col="named_entity_vector_mean",
            target_title=self.target_title2,
            label=self.target_title2_label,
        )
