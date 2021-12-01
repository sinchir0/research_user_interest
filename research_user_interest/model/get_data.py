import glob
import os
import tarfile

import luigi
import pandas as pd
import wget
from research_user_interest.utils.template import GokartTask


class GetTextfileTask(GokartTask):
    textfile_url: str = luigi.Parameter()

    def run(self):
        wget.download(self.textfile_url)
        self.dump("get text file task")


class ExtractTextfileTask(GokartTask):

    tarfile_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()

    def requires(self):
        return GetTextfileTask()

    def run(self):
        # 解凍
        tar = tarfile.open(self.tarfile_path, "r:gz")
        tar.extractall(self.output_path)
        tar.close()

        self.dump("extract textfile task")


class ExtractMainTextTask(GokartTask):

    textfile_path: str = luigi.Parameter()

    def requires(self):
        return ExtractTextfileTask()

    def extract_categories(self):
        # カテゴリーのフォルダのみを抽出
        categories = [
            name
            for name in os.listdir(self.textfile_path)
            if os.path.isdir(f"{self.textfile_path}/{name}")
        ]

        return categories

    def extract_text(self, file_name):
        with open(file_name) as text_file:
            # 今回はタイトル行は外したいので、3要素目以降の本文のみ使用
            title_text = text_file.readlines()[2:]
            title = title_text[0].strip()
            text = title_text[1:]

            # titleの前処理
            title = title.translate(
                str.maketrans({"\n": "", "\t": "", "\r": "", "\u3000": ""})
            )

            # 3要素目以降にも本文が入っている場合があるので、リストにして、後で結合させる
            text = [sentence.strip() for sentence in text]  # 空白文字(スペースやタブ、改行)の削除
            text = list(filter(lambda line: line != "", text))
            text = "".join(text)
            text = text.translate(
                str.maketrans({"\n": "", "\t": "", "\r": "", "\u3000": ""})
            )  # 改行やタブ、全角スペースを消す
        return title, text

    def run(self):

        # リストに前処理した本文と、カテゴリーのラベルを追加していく
        list_title = []
        list_body = []
        list_label = []

        for cat in self.extract_categories():
            text_files = glob.glob(f"{self.textfile_path}/{cat}/*.txt")

            # 前処理extract_main_txtを実施して本文を取得
            text_list = [
                (self.extract_text(text_file)[0], self.extract_text(text_file)[1])
                for text_file in text_files
            ]

            title = [text[0] for text in text_list]
            body = [text[1] for text in text_list]

            label = [cat] * len(body)  # 文の数だけカテゴリー名のラベルのリストを作成

            list_title.extend(title)
            list_body.extend(body)
            list_label.extend(label)

        title_body_label_list = list(zip(list_title, list_body, list_label))

        self.dump(title_body_label_list)


class MakeTextLabelDataFrameTask(GokartTask):
    def requires(self):
        return ExtractMainTextTask()

    def run(self):
        title_body_label_list = self.load()

        title = [text[0] for text in title_body_label_list]
        body = [text[1] for text in title_body_label_list]
        label = [text[2] for text in title_body_label_list]

        df = pd.DataFrame({"title": title, "body": body, "label": label})

        self.dump(df)
