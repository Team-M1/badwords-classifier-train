import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader

from torchsampler import ImbalancedDatasetSampler


def get_data_loaders(
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int = 64,
    return_loader: bool = True,
    use_imbalanced: bool = True,
):
    r"""토크나이저를 입력하면 해당 토크나이저로 인코딩 된 DataLoader를 반환하는 함수
    args:
        tokenizer: 사용할 토크나이저
        batch_size: int: 배치 사이즈, 기본값=64,
        return_loader: bool: True면 DataLoader를 반환하고, False면 Dataset을 반환합니다.
        use_imbalanced: bool: ImbalancedDatasetSampler를 사용할 지의 여부, 기본값=True

    return:
        train_loader, val_loader, test_loader: Tuple[DataLoader]
        """

    def encode(data):
        return tokenizer(data["content"], padding="max_length", truncation=True)

    all_data = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "val": "data/val.csv",
            "test": "data/test.csv",
        },
    )

    all_data = all_data.map(encode)
    all_data.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    if not return_loader:
        return (
            all_data["train"].remove_columns("content"),
            all_data["val"].remove_columns("content"),
            all_data["test"].remove_columns("content"),
        )

    def get_label(dataset):
        return dataset["labels"]

    if use_imbalanced:
        im_sampler = ImbalancedDatasetSampler(
            all_data["train"], callback_get_label=get_label,
        )
        train_loader = DataLoader(
            all_data["train"], batch_size=batch_size, shuffle=True, sampler=im_sampler
        )
    else:
        train_loader = DataLoader(
            all_data["train"], batch_size=batch_size, shuffle=True
        )

    val_loader = DataLoader(all_data["val"], batch_size=batch_size)
    test_loader = DataLoader(all_data["test"], batch_size=batch_size)

    return train_loader, val_loader, test_loader
