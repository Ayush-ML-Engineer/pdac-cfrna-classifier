from pdac_cfrna.config import load_dataset_config
from pdac_cfrna.data import load_dataset

data = ["moore2025_cfnorm", "moore2025_raw"]

for name in data:
    ds = load_dataset(load_dataset_config(name))

    print("\n", name, ds.X.shape, ds.y.value_counts().to_dict())
    print(
        "train samples CEDAR cohort:",
        int(ds.train_mask.sum()),
        "\ntrain samples PDAC vs Healthy count:",
        ds.y[ds.train_mask].value_counts().to_dict(),
    )

    print(
        "\nvalidation samples BCC cohort:",
        int(ds.validation_mask.sum()),
        "\nvalidation samples PDAC vs Healthy count:",
        ds.y[ds.validation_mask].value_counts().to_dict(),
    )
