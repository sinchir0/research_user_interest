import os

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_user_interest.model.extract_text_info import AddVectorTask
from research_user_interest.utils.template import GokartTask
from sklearn.manifold import TSNE


class MakeTsneTask(GokartTask):
    figure_path: str = luigi.Parameter()

    def requires(self):
        return AddVectorTask()

    def delete_null_vector(self, df, col: str):
        """vectorが空のdataframeを除く"""
        df["shape"] = [vector.shape for vector in df[col]]
        df = df[~(df["shape"] == ())].reset_index(drop=True)
        df = df.drop("shape", axis=1)
        return df

    def make_tsne(self, df, col: str):
        df = self.delete_null_vector(df, col)

        vector = np.stack(df[col])

        vector_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(vector)

        # https://qiita.com/g-k/items/120f1cf85ff2ceae4aba
        df_figure = pd.concat(
            [
                df[["label"]],
                pd.DataFrame(vector_embedded, columns=["tsne_1", "tsne_2"]),
            ],
            axis=1,
        )

        label_list = df_figure["label"].unique()

        colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
        plt.figure(figsize=(30, 30))
        for idx, label in enumerate(label_list):
            df_onelabel = df_figure[df_figure["label"] == label]
            plt.scatter(
                df_onelabel["tsne_1"],
                df_onelabel["tsne_2"],
                label=label,
                color=colors[idx],
            )

        plt.legend(fontsize=30)

        os.makedirs(self.figure_path, exist_ok=True)
        plt.savefig(f"{self.figure_path}/{col}_tsne.png", bbox_inches="tight")

    def run(self):
        df = self.load()

        self.make_tsne(df, "noum_vector_mean")
        self.make_tsne(df, "named_entity_vector_mean")

        self.dump("make tsne task")
