import os
import csv
import time
import requests

DL = r"E:\Users\21081\Desktop\Work\Thesis\custom_dataset\downloads"
MANIFEST = os.path.join(DL, "manifest.csv")

S = requests.Session()
S.headers.update({"User-Agent": "doubao-data-scout/2.0 (research non-commercial)"})

ZENODO = [
    ("云杉_NIBIO_tree_species", 14712199, "ObjectDetection_TreeSpecies.zip", "GPL v3 / AGPL v3"),
    ("玉米_USU-Corn-WeedDB",    20044178, "USU-Corn-WeedDB.zip",           "CC BY 4.0"),
]

EXT_SOURCES = [
]

def zenodo_url(rec_id, fname):
    return f"https://zenodo.org/records/{rec_id}/files/{fname}?download=1"

def download(url, dest, timeout=300):
    with S.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dest + ".part"
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
        os.replace(tmp, dest)
        print()
    return dest

def main():
    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as mf:
        w = csv.writer(mf)
        w.writerow(["名称", "来源", "下载URL", "许可", "本地路径", "状态"])
        mf.flush()
        for name, rec_id, fname, lic in ZENODO:
            dest = os.path.join(DL, fname)
            url = zenodo_url(rec_id, fname)
            print(f"[{name}] 开始下载: {url}")
            try:
                if os.path.exists(dest):
                    print(f"  已存在，跳过: {dest}")
                    w.writerow([name, "Zenodo", url, lic, dest, "已存在"])
                    continue
                download(url, dest)
                w.writerow([name, "Zenodo", url, lic, dest, "完成"])
                print(f"[{name}] 完成 -> {dest}")
            except Exception as e:
                w.writerow([name, "Zenodo", url, lic, dest, f"失败: {e}"])
                print(f"[{name}] 失败: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
