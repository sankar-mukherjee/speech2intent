import argparse

from transformers import pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="This was a masterpiece.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="text_intent_classifer/trained_models/100",
    )
    args = parser.parse_args()


    # inference
    text = args.text
    model_dir = args.model_dir
    classifier = pipeline("sentiment-analysis", model=model_dir)
    lables = classifier(text)

    print(lables)
