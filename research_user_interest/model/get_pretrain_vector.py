import os
import tarfile

import luigi
import wget
from research_user_interest.utils.template import GokartTask


class GetPretarinedVectorTask(GokartTask):
    pretrained_site_url: str = luigi.Parameter()
    output_path: str = luigi.Parameter()

    def run(self):
        os.makedirs(self.output_path, exist_ok=True)
        wget.download(self.pretrained_site_url, self.output_path)

        self.dump("get pretrained vector task")


class ExtractPretrainedVectorTask(GokartTask):
    file_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()

    def requires(self):
        return GetPretarinedVectorTask()

    def run(self):
        # 解凍
        tar = tarfile.open(self.file_path, "r:bz2")
        tar.extractall(self.output_path)
        tar.close()

        self.dump("extract pretrainedfile task")
