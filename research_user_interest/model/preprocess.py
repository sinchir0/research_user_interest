from research_user_interest.model.get_data import MakeTextLabelDataFrameTask
from research_user_interest.utils.template import GokartTask
from research_user_interest.utils.text_preprocess import clean_text


class CleanBodyTask(GokartTask):
    def requires(self):
        return MakeTextLabelDataFrameTask()

    def run(self):
        df = self.load()

        df["clean_body"] = df["body"].apply(lambda x: clean_text(x))

        self.dump(df)
