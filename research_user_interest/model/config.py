import luigi


class MasterConfig(luigi.Config):
    model_path: str = luigi.Parameter()
    embedding_dim: int = luigi.IntParameter()
    target_title1: str = luigi.Parameter()
    target_title2: str = luigi.Parameter()
    target_title1_label: str = luigi.Parameter()
    target_title2_label: str = luigi.Parameter()
