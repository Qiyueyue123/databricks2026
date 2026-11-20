"""Parse the scorer's byte-oriented CSV dialect before loading DuckDB."""

from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


EVENTS = ("event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents")
CLICKS = ("click_id", "user_id", "campaign_id", "ts")
PRODUCTS = ("product_id", "category")
CAMPAIGNS = ("campaign_id", "channel")


def rows(files, width):
    for path in files:
        with path.open("rb") as source:
            next(source, None)
            for line in source:
                fields = line.removesuffix(b"\n").split(b",")
                if len(fields) != width:
                    continue
                if b'"' in line:
                    for i, field in enumerate(fields):
                        if len(field) >= 2 and field[:1] == field[-1:] == b'"':
                            field = field[1:-1]
                            fields[i] = field
                        if b'"' in field:
                            break
                    else:
                        yield fields
                else:
                    yield fields


def write_row(target, fields):
    # Quote carriage returns so DuckDB preserves them as field content.
    line = b",".join(fields)
    if b"\r" in line:
        fields = [b'"' + field + b'"' if b"\r" in field else field for field in fields]
        line = b",".join(fields)
    target.write(line + b"\n")


def normalize(source_files, target_path, names):
    with target_path.open("wb") as target:
        write_row(target, [name.encode() for name in names])
        for fields in rows(source_files, len(names)):
            if fields[0]:
                write_row(target, fields)


def write_identity_map(files, target_path):
    ids = {}
    parent = array("I")
    size = array("I")
    minimum = []

    def intern(value):
        index = ids.get(value)
        if index is None:
            index = len(parent)
            ids[value] = index
            parent.append(index)
            size.append(1)
            minimum.append(value)
        return index

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in rows(files, 2):
        if not a or not b or a == b:
            continue
        left, right = root(intern(a)), root(intern(b))
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
        minimum[left] = min(minimum[left], minimum[right])

    with target_path.open("wb") as target:
        target.write(b"source_id,person_id\n")
        for value, index in ids.items():
            person = minimum[root(index)]
            if value != person:
                write_row(target, (value, person))


def prepare(data: Path, scratch: Path, threads: int):
    normalize([data / "products.csv"], scratch / "products.csv", PRODUCTS)
    normalize([data / "campaigns.csv"], scratch / "campaigns.csv", CAMPAIGNS)
    inputs = [(kind, names, list((data / kind).glob("*.csv")))
              for kind, names in (("events", EVENTS), ("clicks", CLICKS))]
    sizes = {path: path.stat().st_size for _, _, files in inputs for path in files}
    links = (data / "links").glob("*.csv")
    identity_path = scratch / "identity.csv"

    if threads < 3 or sum(sizes.values()) < 8 * 1024**2:
        for kind, names, files in inputs:
            normalize(files, scratch / f"{kind}-0.csv", names)
        write_identity_map(links, identity_path)
        return

    # Process large files first so smaller files can fill the remaining slots.
    jobs = []
    for kind, names, files in inputs:
        if not files:
            normalize([], scratch / f"{kind}-0.csv", names)
        for i, path in enumerate(files):
            jobs.append((path, scratch / f"{kind}-{i}.csv", names))
    jobs.sort(key=lambda job: sizes[job[0]], reverse=True)

    with ProcessPoolExecutor(max_workers=min(threads - 1, len(jobs))) as pool:
        pending = [pool.submit(normalize, [path], target, names)
                   for path, target, names in jobs]
        write_identity_map(links, identity_path)
        for job in pending:
            job.result()
