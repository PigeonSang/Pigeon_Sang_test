# -*- coding: utf-8 -*-

import os
import csv
import time
import requests

BASE = r"E:\Users\21081\Desktop\Work\Thesis\custom_dataset"
LOG = os.path.join(BASE, "crawl_log.csv")
MAX_PER_SPECIES = 150

SPECIES = [
    ("Betula platyphylla",   "白桦",   "Betula_platyphylla_白桦"),
    ("Pinus koraiensis",     "红松",   "Pinus_koraiensis_红松"),
    ("Phragmites australis", "芦苇",   "Phragmites_australis_芦苇"),
    ("Setaria viridis",      "狗尾草", "Setaria_viridis_狗尾草"),
    ("Taraxacum officinale", "蒲公英", "Taraxacum_officinale_蒲公英"),
]

S = requests.Session()
S.headers.update({"User-Agent": "doubao-data-scout/1.0 (research non-commercial)"})

def get_taxon_id(name):
    r = S.get("https://api.inaturalist.org/v1/taxa", params={"q": name}, timeout=30)
    j = r.json()
    for hit in j.get("results", []):
        if (hit.get("name") or "").lower() == name.lower():
            return hit["id"], hit.get("preferred_common_name")
    return (j["results"][0]["id"], j["results"][0].get("preferred_common_name")) if j.get("results") else (None, None)

def iter_photos(taxon_id, max_n):
    got = 0
    page = 1
    while got < max_n:
        r = S.get("https://api.inaturalist.org/v1/observations",
                  params={"taxon_id": taxon_id, "per_page": 200, "page": page,
                          "quality_grade": "research", "photos": "true"},
                  timeout=30)
        j = r.json()
        results = j.get("results", [])
        if not results:
            break
        for ob in results:
            if got >= max_n:
                break
            for ph in ob.get("photos", []):
                if got >= max_n:
                    break
                url = (ph.get("url") or "").replace("/square.", "/medium.")
                lic = ph.get("license") or ob.get("license_code") or "unknown"
                obs_id = ob.get("id")
                yield url, lic, obs_id
                got += 1
        page += 1
        time.sleep(0.2)
        if page > 50:
            break

def main():
    with open(LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["学名", "中文名", "taxon_id", "iNat观察ID", "图片URL", "许可", "本地文件"])
        f.flush()
        for name, cn, subdir in SPECIES:
            tid, common = get_taxon_id(name)
            print(f"[{cn}] taxon_id={tid} (common={common})")
            if not tid:
                continue
            d = os.path.join(BASE, subdir)
            os.makedirs(d, exist_ok=True)
            n = 0
            for url, lic, obs_id in iter_photos(tid, MAX_PER_SPECIES):
                try:
                    ir = S.get(url, timeout=30)
                    if ir.status_code != 200:
                        continue
                    ext = ".jpg"
                    fn = os.path.join(d, f"{cn}_{obs_id}_{n}{ext}")
                    with open(fn, "wb") as fp:
                        fp.write(ir.content)
                    w.writerow([name, cn, tid, obs_id, url, lic, fn])
                    n += 1
                except Exception as e:
                    print("  skip", url, e)
            print(f"  -> 下载 {n} 张")
            time.sleep(0.5)

if __name__ == "__main__":
    main()
