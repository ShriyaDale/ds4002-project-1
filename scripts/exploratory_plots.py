#!/usr/bin/env python3
"""Exploratory plots for the airlines reviews dataset.

Creates two plots saved to `output/`:
- `reviews_per_airline.png` — number of reviews per airline
- `customers_by_class.png` — counts of customers mapped to Business, Economy, or Other

Usage:
    python scripts/exploratory_plots.py [--show] [--csv PATH]
"""
from pathlib import Path
import argparse

# parse --show early so we can set a GUI backend before pyplot is imported
early = argparse.ArgumentParser(add_help=False)
early.add_argument("--show", action="store_true")
early_args, _ = early.parse_known_args()

if early_args.show:
    try:
        import matplotlib

        matplotlib.use("TkAgg")
    except Exception:
        # if setting the GUI backend fails, continue with default
        pass

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(show: bool = False, csv_path: str = "data/airlines_reviews.csv"):
    csv = Path(csv_path)
    df = pd.read_csv(csv)

    out = Path("output")
    out.mkdir(exist_ok=True)

    if "Airline" not in df.columns:
        raise KeyError("missing 'Airline' column")

    counts = df["Airline"].value_counts()
    plt.figure(figsize=(10, max(4, len(counts) * 0.25)))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    plt.xlabel("Number of reviews")
    plt.ylabel("Airline")
    plt.title("Reviews per Airline")
    plt.tight_layout()
    a = out / "reviews_per_airline.png"
    plt.savefig(a, dpi=150)
    if show:
        plt.show()
    plt.close()

    if "Class" not in df.columns:
        raise KeyError("missing 'Class' column")

    def cls_label(x):
        if isinstance(x, str):
            lx = x.lower()
            if "business" in lx:
                return "Business"
            if "econom" in lx:
                return "Economy"
        return "Other"

    cls_counts = df["Class"].apply(cls_label).value_counts()
    cls_counts = cls_counts.reindex(["Business", "Economy", "Other"]).fillna(0)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=cls_counts.index, y=cls_counts.values, palette="magma")
    plt.xlabel("Class")
    plt.ylabel("Number of customers")
    plt.title("Customers by Class (Business / Economy / Other)")
    for i, v in enumerate(cls_counts.values):
        plt.text(i, v + max(cls_counts.values) * 0.01, str(int(v)), ha="center")
    plt.tight_layout()
    b = out / "customers_by_class.png"
    plt.savefig(b, dpi=150)
    if show:
        plt.show()
    plt.close()

    print("Saved plots:", a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--csv", default="data/airlines_reviews.csv", help="Path to the CSV file")
    args = parser.parse_args()
    main(show=args.show, csv_path=args.csv)
