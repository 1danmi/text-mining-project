import csv
import json

from prompts import *
from llm_service import llm_service
from sentence_transformer_service import SentenceTransformerService
from datasets import get_spam_dataset, sample_dataset, get_imdb_dataset, get_r8_dataset, get_amazon_dataset

GET_DATASET = {"imdb": get_imdb_dataset, "spam": get_spam_dataset, "amazon": get_amazon_dataset, "r8": get_r8_dataset}

EXAMPLE_TEMPLATE = {"imdb": IMDB_EXAMPLE, "spam": SPAM_EXAMPLE, "amazon": AMAZON_EXAMPLE, "r8": R8_EXAMPLE}

PROMPTS = {
    "imdb": IMDB_PROMPT_TEMPLATE,
    "imdb_simple": IMDB_SIMPLE_PROMPT_TEMPLATE,
    "spam": SPAM_PROMPT_TEMPLATE,
    "spam_simple": SPAM_SIMPLE_PROMPT_TEMPLATE,
    "amazon": AMAZON_PROMPT_TEMPLATE,
    "amazon_simple": AMAZON_SIMPLE_PROMPT_TEMPLATE,
    "r8": R8_PROMPT_TEMPLATE,
    "r8_simple": R8_SIMPLE_PROMPT_TEMPLATE,
}


def detect_label(response: str, labels: set[str]):
    response = response.lower()
    indices = {label: response.find(label) for label in labels if response.find(label) != -1}
    if indices:
        return min(indices, key=indices.get)

    return "unknown"


def run_llm(
    model_path: str,
    model_name: str,
    dataset: dict[str, str],
    task_name: str,
    prompt_template: str,
    labels: set[str],
    num_of_shot: int = 0,
    example_template: str = None,
):
    service = llm_service(model_path)

    dataset_length = len(dataset)
    results_file = f"results/{task_name}_{model_name}_results.csv"
    with open(results_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Ensure proper quoting
        writer.writerow(["text", "label", "response", "prediction"])

    counts = (
        {
            "count": 0,
            "correct": 0,
            "predicted_unknown": 0,
        }
        | {label: 0 for label in labels}
        | {f"predicted_{label}": 0 for label in labels}
    )

    sentence_transformer_service = SentenceTransformerService(dataset)

    for text, label in dataset.items():
        counts["count"] += 1
        if counts["count"] % 100 == 0:
            if counts["correct"] / counts["count"] < 0.5:
                print(
                    f"Stopped with {model_name} after {counts['count']} iterations because accuracy is {counts['correct'] / counts['count']}"
                )
                break
        print(f"Text {counts["count"]}/{dataset_length}")
        counts[label] += 1
        examples = sentence_transformer_service.find_top_n_matches(text, num_of_shot)
        examples_text = "".join(example_template.format(text=example, label=dataset[example]) for example in examples)
        token_limit = 500 - len(prompt_template.split(" ")) - len(examples_text.split(" "))
        if token_limit < 10:
            continue
        dataset_text = " ".join(text.split(" ")[:token_limit])
        response = service.generate(
            prompt_template.format(examples=examples_text, dataset_text=dataset_text), stop=["Text"]
        )
        predicted_label = detect_label(response, labels)
        counts[f"predicted_{predicted_label}"] += 1
        if predicted_label == label:
            counts["correct"] += 1

        with open(results_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Ensure proper quoting
            writer.writerow([text, label, response, predicted_label])
        print(counts)
        print(f"Accuracy: {counts['correct'] / counts['count']}")

    del service
    with open(f"results/{task_name}_{model_name}_final_results.csv", mode="w") as file:
        json.dump(counts, file, indent=4)

    return counts


if __name__ == "__main__":
    # models = ["model/Hermes-3-Llama-3.1-8B.Q8_0.gguf", "model/mistral-7b-v0.1.Q8_0.gguf"]
    # Bad models:
    # Llama3-8B-1.58-100B-tokens-TQ1_0-F16.gguf
    # gpt-neox-20b-q4_0.gguf
    # DeepSeek-R1-Distill-Llama-70B-Q5_K_S.gguf
    # ("model/Meta-Llama-3-8B-Instruct.Q8_0.gguf", "Meta-llama-3-8b-instruct"),
    # ("model/llama-2-7b.Q4_K_M.gguf", "llama-2-7b"),

    models = [
        ("models/Qwen2.5-7B-Instruct.Q8_0.gguf", "Qwen-7b-instruct"),
        ("models/Hermes-3-Llama-3.1-8B.Q8_0.gguf", "hermes-3-llama-3.1-8b"),
        ("models/Mistral-7B-Instruct-v0.3.Q8_0.gguf", "mistral-7b-instruct"),
    ]
    datasets = [
        ("amazon", get_amazon_dataset),
        ("imdb", get_imdb_dataset),
        ("spam", get_spam_dataset),
        ("r8", get_r8_dataset),
    ]

    for dataset_name, get_dataset_func in datasets:
        print("Loading dataset...")
        full_dataset = get_dataset_func()
        labels = set(full_dataset.values())
        dataset = sample_dataset(full_dataset, 5000 // len(labels))
        for model, model_name in models:
            print(f"Running model {model}")
            run_llm(
                model_path=model,
                model_name=model_name,
                dataset=dataset,
                task_name=f"{dataset_name}_one_shot",
                prompt_template=PROMPTS[dataset_name],
                labels=labels,
                example_template=EXAMPLE_TEMPLATE[dataset_name],
                num_of_shot=1,
            )
            run_llm(
                model_path=model,
                model_name=model_name,
                dataset=dataset,
                task_name=f"{dataset_name}_simple_one_shot",
                prompt_template=PROMPTS[f"{dataset_name}_simple"],
                labels=set(dataset.values()),
                example_template=EXAMPLE_TEMPLATE[dataset_name],
                num_of_shot=1,
            )
