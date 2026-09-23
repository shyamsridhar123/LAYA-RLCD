"""Ask the running local service a single typed question. No key in this client."""
import argparse
import json
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=("laya", "decoder", "jev"), required=True)
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
state = "Health 24/100. Ammo 8. Hostiles remaining 3. Medkits remaining 1. Ammo crates remaining 1. Carrying reactor core: no."
request = Request(f"http://127.0.0.1:{args.port}/api/decision",
                  data=json.dumps({"mode": args.model, "state": state}).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=100) as response:
    print(json.dumps(json.load(response), indent=2))
