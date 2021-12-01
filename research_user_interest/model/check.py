from logging import getLogger

import gensim
import luigi
from gokart.config_params import inherits_config_params
from research_user_interest.model.config import MasterConfig
from research_user_interest.model.extract_text_info import AddNamedEntityColTask
from research_user_interest.model.get_data import MakeTextLabelDataFrameTask
from research_user_interest.model.get_pretrain_vector import GetPretarinedVectorTask
from research_user_interest.utils.template import GokartTask

logger = getLogger(__name__)


class CheckNewsTitleTask(GokartTask):
    def requires(self):
        return MakeTextLabelDataFrameTask()

    def run(self):
        df = self.load()

        logger.info(df["title"][df["title"].str.contains("アベンジャーズ")])
        logger.info(df["title"][df["title"].str.contains("Google")])

        self.dump("check news title task")


@inherits_config_params(MasterConfig)
class CheckNewsNoumNamedEntityColTask(GokartTask):
    target_title1: str = luigi.Parameter()
    target_title2: str = luigi.Parameter()

    def requires(self):
        return AddNamedEntityColTask()

    def run(self):
        df = self.load()

        df_extract_1 = df[df["title"].str.contains(self.target_title1)].reset_index(
            drop=True
        )

        logger.info(df_extract_1["body"].values[0])
        logger.info(df_extract_1["clean_body_parse_noum"].values[0])
        logger.info(df_extract_1["clean_body_parse_named_entity"].values[0])

        df_extract_2 = df[df["title"].str.contains(self.target_title2)].reset_index(
            drop=True
        )

        logger.info(df_extract_2["body"].values[0])
        logger.info(df_extract_2["clean_body_parse_noum"].values[0])
        logger.info(df_extract_2["clean_body_parse_named_entity"].values[0])

        self.dump("check news noum named entity col task")


@inherits_config_params(MasterConfig)
class CheckPretrainedVectorTask(GokartTask):
    model_path: str = luigi.Parameter()

    def requires(self):
        return GetPretarinedVectorTask()

    def run(self):
        model = gensim.models.KeyedVectors.load_word2vec_format(
            self.model_path, binary=True
        )

        logger.info(model["アベンジャーズ"])
        logger.info(model["Google"])

        self.dump("check pretrained vector task")
