import io
import os
import zipfile
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".ppm", ".bmp")
test_mixed_path = r'/home/nct/nct01150/dl_openset/data'

def _load_image_from_zip(zip_path, filename, img_size):
    # Decode TensorFlow bytes → Python str
    if isinstance(zip_path, bytes):
        zip_path = zip_path.decode("utf-8")
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(filename) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            img = img.resize(img_size)
            return np.asarray(img, dtype=np.float32) / 255.0


def _tf_load_image(zip_path, filename, label, img_size):
    img = tf.numpy_function(
        _load_image_from_zip,
        [zip_path, filename, img_size],
        tf.float32,
    )
    img.set_shape((*img_size, 3))
    return img, label


def _collect_files(zip_path, split):
    files = []
    labels = []

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            if (
                len(parts) == 3
                and parts[0] == split
                and parts[2].lower().endswith(IMG_EXTENSIONS)
            ):
                files.append(name)
                labels.append(int(parts[1]))

    return np.array(files), np.array(labels)


def _collect_files_from_folder(root_dir, split):
    """
    Supports two structures:
    1. root/<split>/<class_id>/*.png
    2. root/<split>/*.png  (all images same class, e.g., test)
    """
    files = []
    labels = []

    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[INFO] Split directory does not exist: {split_dir}")
        return np.array(files), np.array(labels)

    # Check if split_dir contains subfolders
    subfolders = [d for d in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, d))]
    if subfolders:
        # Standard per-class folder structure
        for class_idx, class_name in enumerate(subfolders):
            class_dir = os.path.join(split_dir, class_name)
            try:
                class_id = int(class_name)
            except Exception:
                class_id = class_idx

            img_count = 0
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    files.append(os.path.join(class_dir, fname))
                    labels.append(class_id)
                    img_count += 1
            print(f"[INFO] Collected {img_count} images for class '{class_name}' (ID={class_id})")
    else:
        # All images directly in split_dir (single pseudo-class)
        img_count = 0
        for fname in os.listdir(split_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                files.append(os.path.join(split_dir, fname))
                labels.append(-1)  # Use -1 for unknown/unlabeled class
                img_count += 1
        print(f"[INFO] Collected {img_count} test images (all labeled -1)")

    print(f"[INFO] Total files collected: {len(files)}")
    return np.array(files), np.array(labels)

def _is_extracted(root_dir):
    return (
        os.path.isdir(os.path.join(root_dir, "train"))
        and os.path.isdir(os.path.join(root_dir, "test"))
    )

def build_datasets_from_zip(zip_path, img_size=(64, 64), batch_size=64, val_split=0.2, seed=42, extract_to=None, path_unseen=None):
    # If extract_to is provided, extract archive only if needed
    if extract_to is not None:
        extract_to = os.path.abspath(extract_to)

        if not _is_extracted(extract_to):
            print(f"[INFO] Extracting dataset to {extract_to} ...")
            os.makedirs(extract_to, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
        else:
            print(f"[INFO] Using existing extracted dataset at {extract_to}")

        # Build from folder
        train_files, train_labels = _collect_files_from_folder(extract_to, "train")
        # if path_unseen, download dataset with seen and unseen samples
        if path_unseen is None:
            test_files, test_labels = _collect_files_from_folder(test_mixed_path, "data_mix")
        # else, load unseen dataset with 32 samples
        else:
            test_files, test_labels = _collect_files_from_folder(path_unseen, "test")

    else:
        train_files, train_labels = _collect_files(zip_path, "train")
        # if path_unseen, download dataset with seen and unseen samples
        if path_unseen is None:
            test_files, test_labels = _collect_files_from_folder(test_mixed_path, "data_mix")
        #  else, load unseen dataset with 32 samples
        else:
            print("[INFO] Loading unseen test set from separate folder...")
            test_files, test_labels = _collect_files_from_folder(path_unseen, "test")

    # Train / Val split
    train_f, val_f, train_y, val_y = train_test_split(
        train_files,
        train_labels,
        test_size=val_split,
        stratify=train_labels,
        random_state=seed,
    )

    # Build tf.data 
    def make_ds(files, labels, shuffle):
        files = tf.constant(files, dtype=tf.string)
        labels = tf.constant(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((files, labels))
        if shuffle:
            ds = ds.shuffle(len(files), seed=seed)
        if extract_to is not None:
            def _read_file(path, label):
                img = tf.io.read_file(path)
                # decode_image may return tensors with unknown shape; ensure static shape before resize
                img = tf.io.decode_image(img, channels=3, expand_animations=False)
                img.set_shape([None, None, 3])
                img = tf.image.resize(img, list(img_size))
                img = tf.cast(img, tf.float32) / 255.0
                img.set_shape((*img_size, 3))
                return img, label

            ds = ds.map(_read_file, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            # Use numpy_function to load from zip (legacy behavior)
            ds = ds.map(
                lambda f, y: _tf_load_image(zip_path, f, y, img_size),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_f, train_y, shuffle=True)
    val_ds = make_ds(val_f, val_y, shuffle=False)
    test_ds = make_ds(test_files, test_labels, shuffle=False)

    num_classes = len(np.unique(train_labels))

    return train_ds, val_ds, test_ds, len(train_f), len(val_f), len(test_files), num_classes
