from torch.utils.data import DataLoader
from .MushroomDataset import MushroomDataset

def make_loaders(
    train,
    val,
    test,
    train_tfms_mild,
    train_tfms_strong,
    val_tfms,
    class_to_idx,
    CLASSES,
    batch_size,
    num_workers,
    device,
):

    train_ds_mild = MushroomDataset(
        train, transform=train_tfms_mild, class_to_idx=class_to_idx, classes=CLASSES
    )
    train_ds_strong = MushroomDataset(
        train, transform=train_tfms_strong, class_to_idx=class_to_idx, classes=CLASSES
    )
    val_ds = MushroomDataset(
        val, transform=val_tfms, class_to_idx=class_to_idx, classes=CLASSES
    )
    test_ds = MushroomDataset(
        test, transform=val_tfms, class_to_idx=class_to_idx, classes=CLASSES
    )

    pin = device.type == "cuda"
    return {
        "train_loader_mild": DataLoader(
            train_ds_mild,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        "train_loader_strong": DataLoader(
            train_ds_strong,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        "val_loader": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        "test_loader": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
    }
