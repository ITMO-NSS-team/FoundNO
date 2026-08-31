import os
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import hf_hub_download


CHUNK_SIZE = 1024 * 1024


def _target_path(source_config):
    if "path" in source_config:
        return Path(os.path.expandvars(source_config["path"]))

    if "cache_dir" not in source_config or "filename" not in source_config:
        raise ValueError("URL/local source must define either path or cache_dir + filename")

    return Path(os.path.expandvars(source_config["cache_dir"])) / source_config["filename"]


def download_url_to_path(url, path, chunk_size=CHUNK_SIZE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"using cached file: {path}")
        return str(path)

    tmp_path = path.with_suffix(path.suffix + ".part")
    request = Request(url, headers={"User-Agent": "FoundNO-PDEBench-data"})

    downloaded = 0
    with urlopen(request) as response:
        total = int(response.headers.get("Content-Length") or 0)

        with tmp_path.open("wb") as file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                file.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = 100.0 * downloaded / total
                    print(f"\rDownloading {path.name}: {pct:6.2f}% ({downloaded}/{total})", end="")
                else:
                    print(f"\rDownloading {path.name}: {downloaded} bytes", end="")

    print()
    tmp_path.replace(path)
    return str(path)


def get_hf_token(source_config):
    token_env = source_config.get("token_env")
    if token_env is None:
        return None

    token = os.environ.get(token_env)
    if not token:
        print(f"HF token env '{token_env}' is not set; using anonymous download")
        return None

    return token


def get_access_data_path(source_config):
    location = source_config["location"]

    if location == "local":
        path = Path(os.path.expandvars(source_config["path"]))

        if not path.exists():
            raise FileNotFoundError(f"Local data file not found: {path}")

        return str(path)

    if location == "url":
        return download_url_to_path(
            source_config["url"],
            _target_path(source_config),
        )

    if location == "huggingface":
        repo_id = source_config["repo_id"]
        filename = source_config["filename"]
        cache_dir = source_config.get("cache_dir")

        token = get_hf_token(source_config)

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=token
        )

        return path

    raise ValueError(f"Unknown source location: {location}")
