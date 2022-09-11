import json
from optparse import Option
import sys
from typing import Optional

import fire  # type: ignore
import pandas as pd


def load(
    load_path: str,
    save_path: str,
    input_col: str,
    output_col: str,
    headers: Optional[list[str]] = None,
    accepted_outputs: Optional[list[str]] = None,
):
    """Transform a csv dataset into a jsonl dataset.

    Example use: python misc/csv_to_jsonl.py my_data/twitter_validation.csv counterfactual_dataset_generator/data/examples/twitter-sentiment.jsonl Tweet_content Sentiment --headers ['Tweet_ID','Entity','Sentiment','Tweet_content'] --accepted_outputs ['Positive','Neutral','Negative']"""
    df = pd.read_csv(load_path, names=headers)
    lines_written = 0
    with open(save_path, "w", encoding="utf-8") as outfile:
        for i, row in df.iterrows():
            if accepted_outputs is None or row[output_col] in accepted_outputs:
                json_dict = {"input": row[input_col], "expected_output": row[output_col]}
                json.dump(json_dict, outfile)
                outfile.write("\n")
                lines_written += 1
    print(f"Done! {lines_written} lines written.")


if __name__ == "__main__":
    fire.Fire(load)