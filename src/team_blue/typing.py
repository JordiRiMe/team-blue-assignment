from typing import Literal

DatasetKind = Literal["training", "validation"]
InvalidTenurePolicy = Literal["drop_rows", "drop_customers", "raise"]
ConflictingTargetPolicy = Literal["drop_customers", "raise"]