"""Acquire a balanced, group-verified PlantVillage subset without cloning the full dataset."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import ssl
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import certifi
import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from leaf_pipeline import ROOT, SEED, file_hash, read_manifest, write_csv, write_json

REVISION = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
REPOSITORY = "https://github.com/spMohanty/PlantVillage-Dataset"
RAW = f"https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/{REVISION}/"
API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/"
PLANTS = ("Apple", "Pepper,_bell", "Tomato")
DISEASES = {
    "Apple": ("Apple_scab", "Cedar_apple_rust"),
    "Pepper,_bell": ("Bacterial_spot",),
    "Tomato": ("Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
               "Septoria_leaf_spot", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus"),
}
LABEL_MAPPING = {f"{plant}___{disease}": "Diseased"
                 for plant, diseases in DISEASES.items() for disease in diseases}
LABEL_MAPPING.update({f"{plant}___healthy": "Healthy" for plant in PLANTS})


def fetch(url):
    context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "COM0418-Leaf-Project",
                                           "Accept": "application/vnd.github+json"})
            with urlopen(request, timeout=60, context=context) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def cached_json(url, path):
    path = Path(path)
    if not path.exists():
        content = fetch(url)
        json.loads(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return json.loads(path.read_text())


def resolve_group(filename, label, mapping):
    key = Path(filename).stem.split("___")[-1].split("copy")[0].strip().lower()
    choices = {item for item in mapping.get(key, []) if item.startswith(label + ":::")
               and item.rsplit(":::", 1)[-1].lower() not in ("nan", "none", "")}
    return next(iter(choices)) if len(choices) == 1 else None


def build_selection(root):
    source = root / "data/source"
    mapping = cached_json(RAW + "leaf_grouping/leaf-map.json", source / "leaf-map.json")
    directories = cached_json(API + f"contents/raw/color?ref={REVISION}",
                              source / "color-directories.json")
    sha_by_label = {item["name"]: item["sha"] for item in directories}
    candidates, exclusions, inventory = [], [], []
    for label, health in sorted(LABEL_MAPPING.items()):
        tree = cached_json(API + "git/trees/" + sha_by_label[label], source / f"{label}.json")
        if tree.get("truncated"):
            raise ValueError(f"Incomplete upstream file listing: {label}")
        grouped = defaultdict(list)
        for node in tree["tree"]:
            if node["type"] != "blob" or Path(node["path"]).suffix.lower() not in (".jpg", ".png", ".jpeg"):
                continue
            group = resolve_group(node["path"], label, mapping)
            if group is None:
                exclusions.append({"source_path": f"raw/color/{label}/{node['path']}",
                                   "reason": "missing_or_ambiguous_leaf_group"})
                continue
            source_path = f"raw/color/{label}/{node['path']}"
            grouped[group].append({"source_path": source_path, "original_label": label,
                                   "health": health, "plant": label.split("___")[0],
                                   "leaf_group": group, "source_blob_sha": node["sha"]})
        inventory.append({"original_label": label, "health": health,
                          "available_images": sum(len(values) for values in grouped.values()),
                          "verified_groups": len(grouped)})
        print(f"{label}: {len(grouped)} verified groups", flush=True)
        # One image per physical leaf makes the sampled observations independent by supplied IDs.
        for group, records in sorted(grouped.items()):
            records.sort(key=lambda row: hashlib.sha256((str(SEED) + row["source_path"]).encode()).hexdigest())
            candidates.append(records[0])
    rng = random.Random(SEED)
    selected = []
    for plant in PLANTS:
        for health in ("Healthy", "Diseased"):
            pools = defaultdict(list)
            for row in candidates:
                if row["plant"] == plant and row["health"] == health:
                    pools[row["original_label"]].append(row)
            for pool in pools.values():
                rng.shuffle(pool)
            sampled = []
            while len(sampled) < 150:
                progressed = False
                for label in sorted(pools):
                    if pools[label] and len(sampled) < 150:
                        sampled.append(pools[label].pop())
                        progressed = True
                if not progressed:
                    raise ValueError(f"Fewer than 150 independent {health} leaves for {plant}")
            selected.extend(sampled)
    write_csv(root / "data/manifests/source_inventory.csv", inventory)
    write_csv(root / "data/manifests/exclusions.csv", exclusions,
              ["source_path", "reason"])
    selected.sort(key=lambda row: row["source_path"])
    for index, row in enumerate(selected, 1):
        row["image_id"] = f"leaf_{index:04d}"
        row["path"] = f"data/raw/{row['original_label']}/{Path(row['source_path']).name}"
        row["source_url"] = RAW + quote(row["source_path"], safe="/")
        row["source_revision"] = REVISION
    write_json(source / "selection.json", selected)
    return selected


def download_image(row, root):
    validate_source_record(row)
    path = root / row["path"]
    if not path.exists():
        content = fetch(row["source_url"])
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != row["source_blob_sha"]:
            raise ValueError(f"Source checksum mismatch: {row['image_id']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o444)
    content = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if blob != row["source_blob_sha"]:
        raise ValueError(f"Local image changed: {row['image_id']}")
    bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or min(bgr.shape[:2]) < 64:
        raise ValueError(f"Image failed quality/decode validation: {row['image_id']}")
    result = dict(row, sha256=hashlib.sha256(content).hexdigest(),
                  pixel_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
                  width=bgr.shape[1], height=bgr.shape[0])
    thumbnail = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(cv2.resize(thumbnail, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)[:8, :8]
    bits = (dct > np.median(dct.reshape(-1)[1:])).reshape(-1)
    phash = int.from_bytes(np.packbits(bits).tobytes(), "big")
    return result, thumbnail, phash


def validate_source_record(row):
    label = row["original_label"]
    name = Path(row["source_path"]).name
    expected = f"raw/color/{label}/{name}"
    if (label not in LABEL_MAPPING or row["source_revision"] != REVISION
            or row["source_path"] != expected or row["path"] != f"data/raw/{label}/{name}"
            or row["source_url"] != RAW + quote(expected, safe="/")):
        raise ValueError("Image source/path differs from the pinned PlantVillage acquisition protocol")


def assign_partitions(records, thumbnails, phashes, root):
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    duplicates = []
    # shortcut: near-duplicate screening assumes similar alignment; add matching for external views.
    for i in range(len(records)):
        for j in range(i):
            exact = records[i]["pixel_sha256"] == records[j]["pixel_sha256"]
            hamming = (phashes[i] ^ phashes[j]).bit_count()
            if not exact and hamming > 6:
                continue
            error = float(np.abs(thumbnails[i].astype(float) - thumbnails[j]).mean())
            if exact or error <= 8.0:
                if records[i]["health"] != records[j]["health"]:
                    raise ValueError("Duplicate-like images have conflicting labels; resolve before splitting")
                parent[find(i)] = find(j)
                duplicates.append({"image_id_a": records[j]["image_id"],
                                   "image_id_b": records[i]["image_id"],
                                   "exact": exact, "phash_distance": hamming, "rgb_mae": error})
    for index, row in enumerate(records):
        row["split_group"] = "group_" + records[find(index)]["image_id"]
    strata = np.array([row["plant"] + ":" + row["health"] for row in records])
    groups = np.array([row["split_group"] for row in records])
    dev, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        np.zeros(len(records)), strata, groups))
    test_indices = set(test)
    for index, row in enumerate(records):
        row["partition"] = "test" if index in test_indices else "development"
        row["cv_fold"] = ""
        row["annotation_partition"] = ""
    for fold, (_, validation) in enumerate(StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=SEED).split(
                np.zeros(len(dev)), strata[dev], groups[dev])):
        for index in dev[validation]:
            records[index]["cv_fold"] = fold
    rng = random.Random(SEED + 1)
    for plant in PLANTS:
        for partition, count in (("development", 2), ("test", 3)):
            for health in ("Healthy", "Diseased"):
                pool = [row for row in records if row["plant"] == plant
                        and row["partition"] == partition and row["health"] == health]
                for row in rng.sample(pool, count):
                    row["annotation_partition"] = partition
    write_csv(root / "data/manifests/duplicate_pairs.csv", duplicates,
              ["image_id_a", "image_id_b", "exact", "phash_distance", "rgb_mae"])
    return records


def validate_dataset(root=ROOT):
    root = Path(root)
    rows = read_manifest(root)
    if len(rows) != 900 or len({row["image_id"] for row in rows}) != 900:
        raise ValueError("Expected 900 unique images")
    for row in rows:
        validate_source_record(row)
        if LABEL_MAPPING.get(row["original_label"]) != row["health"]:
            raise ValueError("Unapproved binary label mapping")
        if file_hash(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['image_id']}")
    for field in ("leaf_group", "split_group", "sha256", "pixel_sha256"):
        memberships = defaultdict(set)
        for row in rows:
            memberships[row[field]].add(row["partition"])
        if any(len(partitions) != 1 for partitions in memberships.values()):
            raise ValueError(f"Partition leakage through {field}")
    for field in ("leaf_group", "split_group", "sha256", "pixel_sha256"):
        memberships = defaultdict(set)
        for row in rows:
            if row["partition"] == "development":
                memberships[row[field]].add(row["cv_fold"])
        if any(len(folds) != 1 for folds in memberships.values()):
            raise ValueError(f"Cross-validation leakage through {field}")
    counts = Counter((row["plant"], row["health"]) for row in rows)
    if set(counts.values()) != {150} or len(counts) != 6:
        raise ValueError("Plant/health sample balance changed")
    annotation_counts = Counter(row["annotation_partition"] for row in rows)
    if annotation_counts["development"] != 12 or annotation_counts["test"] != 18:
        raise ValueError("Reference-mask sampling changed")
    return rows


def prepare(root=ROOT):
    root = Path(root)
    if (root / "data/manifests/images.csv").exists():
        missing = [row for row in read_manifest(root) if not (root / row["path"]).exists()]
        if missing:
            print(f"Restoring {len(missing)} original images from the pinned manifest", flush=True)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for _ in pool.map(lambda row: download_image(row, root), missing):
                    pass
        rows = validate_dataset(root)
        print("Existing dataset verified; fixed partitions preserved.", flush=True)
        return rows
    selection_path = root / "data/source/selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else build_selection(root)
    records, thumbnails, phashes = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, (record, thumbnail, phash) in enumerate(
                pool.map(lambda row: download_image(row, root), selection), 1):
            records.append(record)
            thumbnails.append(thumbnail)
            phashes.append(phash)
            if number % 100 == 0:
                print(f"Downloaded and verified {number}/900 images", flush=True)
    rows = assign_partitions(records, thumbnails, phashes, root)
    write_csv(root / "data/manifests/images.csv", rows)
    write_json(root / "data/manifests/label_mapping.json", {
        "labels": {"Healthy": 0, "Diseased": 1}, "allowed_original_labels": LABEL_MAPPING,
        "exclusions": {"Grape": "No healthy leaf grouping metadata in the selected source revision",
                       "Apple___Black_rot": "Grouping label uses a different name; not resolved for this subset",
                       "Tomato___Tomato_mosaic_virus": "No verified leaf grouping metadata",
                       "Tomato___Spider_mites Two-spotted_spider_mite": "Pest-only category outside binary disease scope"}})
    write_json(root / "data/manifests/dataset_summary.json", {
        "source": REPOSITORY, "revision": REVISION, "seed": SEED,
        "n_images": len(rows), "n_original_leaf_groups": len({r['leaf_group'] for r in rows}),
        "counts": dict(Counter(f"{r['partition']}/{r['plant']}/{r['health']}" for r in rows)),
        "near_duplicate_rule": "pHash Hamming <= 6 and 64x64 RGB MAE <= 8, or identical decoded pixels",
        "sampling": "One image per supplied leaf group; equal disease-subtype round-robin sampling"})
    validate_dataset(root)
    print(json.dumps(Counter(row["partition"] for row in rows)), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify the existing fixed dataset only")
    args = parser.parse_args()
    validate_dataset() if args.verify else prepare()
