import json
import re
import os
import csv
import random

import pandas as pd
import xmltodict

REUTERS_CATEGORIES = {
    "earn": "earnings",
    "acq": "acquisitions",
    "money-fx": "money",
    "crude": "oil",
    "grain": "grain",
    "trade": "trade",
    "interest": "monetary",
    "wheat": "wheat",
    "ship": "shipping"
}


def get_imdb_dataset() -> dict[str, str]:
    with open("datasets/IMDB Dataset/IMDB Dataset.csv", "r", encoding="utf8") as f:
        csv_reader = csv.reader(f)
        return {text.replace("<br >", "").replace("<br />", "").strip(): sentiment for (text, sentiment) in csv_reader}


def get_spam_dataset() -> dict[str, str]:
    with open("datasets/SMS Spam Collection Dataset/spam.csv", "r", encoding="utf8", errors="ignore") as f:
        dataset = {}
        for line in f:
            label, *texts = line.split(",")
            text = " ".join(texts).strip()
            if text:
                dataset[text] = label
        return dataset


def get_amazon_dataset() -> dict[str, str]:
    files = ["test.csv", "train.csv"]
    dataset = {}
    for file in files:
        with open(f"datasets/Amazon Reviews/amazon_review_polarity_csv.tgz/amazon_review_polarity_csv/{file}", "r", encoding="utf8", errors="ignore") as f:
            for line in f:
                label, *texts = line.split(",")
                text = " ".join(texts).strip()
                if text:
                    if label == "\"1\"":
                        dataset[text] = "negative"
                    if label == "\"2\"":
                        dataset[text] = "positive"
    return dataset


def load_reuters_dataset():
    if os.path.exists("datasets/Reuters 21578/dataset.json"):
        with open("datasets/Reuters 21578/dataset.json", "r", encoding="utf8") as f:
            return json.load(f)
    reuters_dataset = []
    DECLARATION = "<!DOCTYPE lewis SYSTEM \"lewis.dtd\">"
    PREFIX = "<LEWIS>"
    SUFFIX = "</LEWIS>"
    for file in os.listdir("datasets/Reuters 21578/reuters21578.tar"):
        if file.endswith(".xml"):
            print(f"Reading file {file}")
            with open(f"datasets/Reuters 21578/reuters21578.tar/{file}", "r", encoding="utf8", errors="ignore") as f:
                xml_data = f.read()
                xml_data = xml_data.replace("&", "&amp;").replace(DECLARATION, DECLARATION + PREFIX) + SUFFIX
                reuters_dataset += xmltodict.parse(xml_data)["LEWIS"]["REUTERS"]

    with open("datasets/Reuters 21578/dataset.json", "w", encoding="utf8") as json_file:
        json.dump(reuters_dataset, json_file, indent=4)

    return reuters_dataset


def pre_process_reuters_dataset():
    dataset = load_reuters_dataset()
    processed_dataset = {}
    for record in dataset:
        topics = record.get("TOPICS")
        if topics:
            text = f"{record.get("TEXT").get("TITLE", "")} {record["TEXT"].get("BODY", "")}".strip()
            if text:
                if isinstance(topics["D"], str):
                    if processed_dataset.get(topics["D"]):
                        processed_dataset[topics["D"]].append(text)
                    else:
                        processed_dataset[topics["D"]] = [text]
                elif isinstance(topics["D"], list):
                    if processed_dataset.get(topics["D"][0]):
                        processed_dataset[topics["D"][0]].append(text)
                    else:
                        processed_dataset[topics["D"][0]] = [text]

    return processed_dataset


def get_top_n_categories(dataset: dict, n: int):
    sorted_keys = sorted(dataset, key=lambda k: len(dataset[k]), reverse=True)
    top_keys = sorted_keys[:n]
    return {k: dataset[k] for k in top_keys}


def flatten_reuters_dataset(dataset: dict):
    flat_dataset = {}
    for key, value in dataset.items():
        for text in value:
            flat_dataset[text] = REUTERS_CATEGORIES[key]

    return flat_dataset


def get_r8_dataset():
    preprocessed_dataset = pre_process_reuters_dataset()
    top_8_reuters = get_top_n_categories(preprocessed_dataset, 8)
    return flatten_reuters_dataset(top_8_reuters)


def sample_dataset(dataset: dict[str, str], n: int) -> dict[str, str]:
    df = pd.DataFrame.from_records(list(dataset.items()), columns=["text", "label"])
    sampled_df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(n=min(len(x), n)))
    sampled_items = list(sampled_df.itertuples(index=False, name=None))
    random.shuffle(sampled_items)

    # Reconstruct shuffled dictionary
    shuffled_dataset = dict(sampled_items)
    return shuffled_dataset


if __name__ == "__main__":
    x = get_amazon_dataset()
