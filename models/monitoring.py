import re
from scipy.stats import ks_2samp


def data_drift_scores(
    train_docs: list[str], val_docs: list[str], new_docs: list[str]
) -> dict[str, float]:
    """Compute how different the input is compared to the validation set.

    The returned scores are computed as the Kolmogorov-Smirnov statistics from two-sided
    tests comparing the distributions of text length and out-of-vocabulary word fraction.
    """
    def clean(text):
        """Strip punctuation and lowercase."""
        return re.sub(r"[^\w\s]", "", text).lower()

    train_vocab = set().union(*(clean(d).split() for d in train_docs))

    def fraction_out_of_vocab(text):
        """% of words in text that were not present in the training set."""
        return sum(1 for w in clean(text).split() if w not in train_vocab) / len(text.split())

    def length(text):
        """Number of words in the text."""
        return len(text.split())

    return {
        metric_name: float(
            ks_2samp([metric(d) for d in val_docs], [metric(d) for d in new_docs]).statistic
        )
        for metric_name, metric in [
            ("length", length), ("fraction_out_of_vocab", fraction_out_of_vocab)
        ]
    }