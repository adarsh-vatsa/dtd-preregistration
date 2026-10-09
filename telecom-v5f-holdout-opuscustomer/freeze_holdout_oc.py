import datetime, hashlib, json, os, random, time, urllib.request
F = "/home/avatsa/dtd-bounded/bounded_v5f_frozen"; H = os.path.join(F, "holdout_oc"); SRC = "/home/avatsa/dtd-bounded/bounded_v5f/hoc"
os.makedirs(H, exist_ok=False)
for f in ["run_holdout_oc.sh", "PREREG_template.json", "freeze_holdout_oc.py"]:
    open(os.path.join(H, f), "wb").write(open(os.path.join(SRC, f), "rb").read())
t0 = datetime.datetime.now(datetime.timezone.utc)
lines = [f"{hashlib.sha256(open(os.path.join(H, f), 'rb').read()).hexdigest()}  {f}" for f in ["run_holdout_oc.sh", "PREREG_template.json", "freeze_holdout_oc.py"]]
lines += [l for l in open(os.path.join(F, "v5f_frozen_sha256.txt")).read().splitlines() if "  " in l]
open(os.path.join(H, "holdout_oc_sha256.txt"), "w").write("\n".join(lines) + "\n" + t0.strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
print("hashed", t0.isoformat())
while True:
    p = json.load(urllib.request.urlopen("https://beacon.nist.gov/beacon/2.0/pulse/last", timeout=20))["pulse"]
    if datetime.datetime.fromisoformat(p["timeStamp"].replace("Z", "+00:00")) >= t0 + datetime.timedelta(seconds=60):
        break
    time.sleep(10)
json.dump(p, open(os.path.join(H, "holdout_oc_beacon_pulse.json"), "w"), indent=1)
sp = json.load(open(os.path.join(F, "telecom_splits.json")))
used = {x.strip() for x in open("/home/avatsa/dtd-bounded/bounded/v5_eval_holdout_ids.txt") if x.strip()}
pool = sorted(set(sp["holdout"]) - used); assert len(pool) == 441, len(pool)
ids = random.Random(int(p["outputValue"][:16], 16)).sample(pool, 60)
open(os.path.join(H, "holdout_oc_ids.txt"), "w").write("\n".join(ids) + "\n")
pr = json.load(open(os.path.join(H, "PREREG_template.json")))
pr["written_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
pr["hashes"] = open(os.path.join(H, "holdout_oc_sha256.txt")).read()
pr["beacon"] = {"uri": p["uri"], "timeStamp": p["timeStamp"], "outputValue": p["outputValue"]}
pr["set"] = {"n": 60, "pool": "441 holdout tasks never used by an earlier evaluation", "rule": "random.Random(int(outputValue[:16],16)).sample(sorted(pool), 60)"}
json.dump(pr, open(os.path.join(H, "PREREG_telecom_v5f_holdout_opuscustomer.json"), "w"), indent=1)
print("pulse", p["timeStamp"], "drawn", len(ids))
