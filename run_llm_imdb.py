import csv

SENTIMENT_PROMPT_TEMPLATE = """This is an overall sentiment classifier for input opinion snippets.
Present CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the sentiment determination of the input.
Next, deduce the diagnostic REASONING process from premises (i.e., clues, input) that support the sentiment determination.
Finally, based on clues, reasoning and the input, categorize the overall SENTIMENT of input as Positive or Negative.
Limit your answer 100 tokens or less.

Text: "{dataset_text}"

SENTIMENT:"""




def is_negative_or_positive(response: str):
    response = response.lower()
    negative_index = response.find("negative")
    positive_index = response.find("positive")
    if negative_index != -1 and positive_index != -1:
        return "negative" if negative_index < positive_index else "positive"

    if negative_index != -1:
        return "negative"
    if positive_index != -1:
        return "positive"
    return "unknown"


def run_llm_imdb(model_path: str, model_name: str):
    service = llm_service(model_path)

    imdb_dataset = get_imdb_dataset()
    length = len(imdb_dataset)
    results_file = f"results/result_{model_name}_imdb.csv"
    with open(results_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Ensure proper quoting
        writer.writerow(["text", "label", "response", "prediction"])

    counts = {
        "count": 0,
        "correct": 0,
        "positive": 0,
        "negative": 0,
        "predicted_positive": 0,
        "predicted_negative": 0,
        "predicted_unknown": 0,
    }
    for text, label in imdb_dataset.items():
        counts["count"] += 1
        if counts["count"] % 100 == 0:
            if counts["correct"] / counts["count"] < 0.5:
                print(
                    f"Stopped with {model_name} after {counts['count']} iterations because accuracy is {counts['correct'] / counts['count']}")
                break
        print(f"Text {counts["count"]}/{length}")
        counts[label] += 1
        response = service.generate(SENTIMENT_PROMPT_TEMPLATE.format(dataset_text=text[:400]), stop=["Text"])
        predicted_label = is_negative_or_positive(response)
        counts[f"predicted_{predicted_label}"] += 1
        if predicted_label == label:
            counts["correct"] += 1

        with open(results_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Ensure proper quoting
            writer.writerow([text, label, response, predicted_label])
        print(counts)
        print(f"Accuracy: {counts['correct'] / counts['count']}")

    del service
