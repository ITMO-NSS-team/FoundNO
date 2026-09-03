import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

# ROOT = Path(r"D:\PDEBench_data")
# PDE_NAME = "diff_sorp"
# FILENAME = "1D_diff-sorp_NA_NA.h5"
# URL = "https://darus.uni-stuttgart.de/api/access/datafile/133020"

# https://darus.uni-stuttgart.de/api/access/datafile/133017

# # get data: 1D_diff-sorp_NA_NA.h5
# https://darus.uni-stuttgart.de/api/access/datafile/133020
#
# # visualize
# python visualize_pdes.py --pde_name "diff_sorp" --data_path "./"

# # get data: ReacDiff_Nu1.0_Rho1.0.hdf5
# https://darus.uni-stuttgart.de/api/access/datafile/133181
#
# # visualize
# python visualize_pdes.py --pde_name "1d_reacdiff"

# # get data: 1D_Advection_Sols_beta0.4.hdf5
# https://darus.uni-stuttgart.de/api/access/datafile/133110
#
# # visualize
# python visualize_pdes.py --pde_name "advection" --param 0.4

# # get data: 1D_Burgers_Sols_Nu0.01.hdf5
# https://darus.uni-stuttgart.de/api/access/datafile/133136
#
# # visualize
# python visualize_pdes.py --pde_name "burgers" --param 0.01

# # get data: 1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5
# https://darus.uni-stuttgart.de/api/access/datafile/135485
#
# # visualize
# python visualize_pdes.py --pde_name "1d_cfd"

# # get data: 2D_diff-react_NA_NA.h5
# https://darus.uni-stuttgart.de/api/access/datafile/133017
#
# # visualize
# python visualize_pdes.py --pde_name "2d_reacdiff"

# # get data: 2D_DarcyFlow_beta1.0_Train.hdf5
# https://darus.uni-stuttgart.de/api/access/datafile/133219
#
# # visualize
# python visualize_pdes.py --pde_name "darcy"

# # get data: 2D_rdb_NA_NA.h5
# https://darus.uni-stuttgart.de/api/access/datafile/133021
#
# # visualize
# python visualize_pdes.py --pde_name "swe" --data_path "./"

ROOT = Path(r"D:\PDEBench_data")
PDE_NAME = "DarcyFlow"
FILENAME = "2D_DarcyFlow_beta0.01_Train.hdf5"
URL = "https://darus.uni-stuttgart.de/api/access/datafile/133217"

MAX_BYTES = 100 * 1024 * 1024  # вот здесь None для скачивания полного файла
CHUNK_SIZE = 1024 * 1024


def md5_file(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    if MAX_BYTES is None:
        out_path = ROOT / FILENAME
    else:
        out_path = ROOT / f"{FILENAME}.first_{MAX_BYTES // (1024 * 1024)}mb"

    print(f"PDE: {PDE_NAME}")
    print(f"URL: {URL}")
    print(f"Output: {out_path}")

    req = Request(URL, headers={"User-Agent": "FoundNO-PDEBench-data"})

    downloaded = 0
    with urlopen(req) as response:
        total = int(response.headers.get("Content-Length") or 0)
        print(f"Remote size: {total} bytes" if total else "Remote size: unknown")

        with out_path.open("wb") as f:
            while True:
                if MAX_BYTES is not None:
                    remaining = MAX_BYTES - downloaded
                    if remaining <= 0:
                        break
                    read_size = min(CHUNK_SIZE, remaining)
                else:
                    read_size = CHUNK_SIZE

                chunk = response.read(read_size)
                if not chunk:
                    break

                f.write(chunk)
                downloaded += len(chunk)

                if MAX_BYTES is not None:
                    pct = 100.0 * downloaded / MAX_BYTES
                    print(f"\rDownloaded part: {pct:6.2f}% ({downloaded}/{MAX_BYTES})", end="")
                elif total:
                    pct = 100.0 * downloaded / total
                    print(f"\rDownloaded full file: {pct:6.2f}% ({downloaded}/{total})", end="")
                else:
                    print(f"\rDownloaded: {downloaded} bytes", end="")

    print()
    print(f"Saved: {out_path}")
    print(f"Size: {out_path.stat().st_size} bytes")
    print(f"MD5 of downloaded part/file: {md5_file(out_path)}")


if __name__ == "__main__":
    main()
