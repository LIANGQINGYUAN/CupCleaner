from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Tuple, Union
import re

def subtokenize_comment(comment_line: str, remove_tag=True) -> str:
    """Subtokenize comments from https://github.com/panthap2/deep-jit-inconsistency-detection/blob/master/data_processing/data_formatting_utils.py"""

    if remove_tag:
        comment_line = remove_tag_string(comment_line)
    comment_line = remove_html_tag(
        comment_line.replace("/**", "")
        .replace("**/", "")
        .replace("/*", "")
        .replace("*/", "")
        .replace("*", "")
        .strip()
    )
    comment_line = re.findall(
        r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.strip()
    )
    comment_line = " ".join(comment_line)
    comment_line = comment_line.replace("\n", " ").strip()

    tokens = comment_line.split(" ")
    subtokens = []
    for token in tokens:
        curr = re.sub("([a-z0-9])([A-Z])", r"\1 \2", token).split()
        try:
            new_curr = []
            for c in curr:
                by_symbol = re.findall(
                    r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip()
                )
                new_curr = new_curr + by_symbol

            curr = new_curr
        except:
            curr = []
        subtokens = subtokens + [c.lower() for c in curr]

    comment_line = " ".join(subtokens)
    return comment_line.lower()

def remove_tag_string(line: str) -> str:
    search_strings = [
        "@return",
        "@ return",
        "@param",
        "@ param",
        "@throws",
        "@ throws",
    ]
    for s in search_strings:
        line = line.replace(s, "").strip()
    return line


def remove_html_tag(line: str):
    SPECIAL_TAGS = [
        "{",
        "}",
        "@code",
        "@docRoot",
        "@inheritDoc",
        "@link",
        "@linkplain",
        "@value",
    ]
    clean = re.compile("<.*?>")
    line = re.sub(clean, "", line)

    for tag in SPECIAL_TAGS:
        line = line.replace(tag, "")

    return line


def compute_bleu_scores(
    references: List[str], hypotheses: List[str], dataset: str,
) -> Tuple[float, List]:
    """Compute BLEU score and return the Tuple[average BLEU, list of bleu]"""

    if "comment-update" in dataset:
        refs = [subtokenize_comment(ref) for ref in references]
        hypos = [subtokenize_comment(hyp) for hyp in hypotheses]
    else:
        refs = references
        hypos = hypotheses
    bleu_4_sentence_scores = []
    for ref, hyp in zip(refs, hypos):
        if hyp == "":
            hyp = "<EMPTY>"
        bleu_4_sentence_scores.append(
            sentence_bleu(
                [ref.split()],
                hyp.split(),
                smoothing_function=SmoothingFunction().method2,
                auto_reweigh=True,
            )
            * 100
        )
    return (
        sum(bleu_4_sentence_scores) / float(len(bleu_4_sentence_scores)),
        bleu_4_sentence_scores,
    )