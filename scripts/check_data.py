from pdac_cfrna.config import load_dataset_config
from pdac_cfrna.data import load_dataset

for name in ("moore2025_cfnorm", "moore2025_raw"):
    ds = load_dataset(load_dataset_config(name))
    print(name, ds.X.shape, ds.y.value_counts().to_dict())
    print(
        "train samples:",
        int(ds.train_mask.sum()),
        "validation samples:",
        int(ds.validation_mask.sum()),
    )
