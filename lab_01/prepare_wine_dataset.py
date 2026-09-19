from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_wine

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()
    output_path = DATA_DIR / "wine_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Файл {output_path} успешно создан!")


if __name__ == "__main__":
    main()
