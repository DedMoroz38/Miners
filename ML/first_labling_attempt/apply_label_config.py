#!/usr/bin/env python3
"""Apply label_config.xml to the running Label Studio project via the REST API.

Reads the admin API token straight from the local Label Studio SQLite DB, then
PATCHes the project so the server recomputes its parsed config + data summary
(safer and more complete than editing the DB by hand). Run while `./run.sh` is up.

    python apply_label_config.py            # project 1 on http://localhost:8080

Fallback if this ever fails: paste label_config.xml into the UI at
Settings -> Labeling Interface -> Code -> Save.
"""
import argparse
import json
import os
import sqlite3
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, ".label_studio", "label_studio.sqlite3")
CONFIG = os.path.join(HERE, "label_config.xml")


def db_token():
    """Return the first API token, enabling legacy-token auth (LS 1.23+ default-off)."""
    if not os.path.exists(DB):
        return None
    con = sqlite3.connect(DB)
    try:
        # LS 1.23 disables legacy `Token` auth by default; re-enable it (idempotent).
        try:
            con.execute("update jwt_auth_jwtsettings set legacy_api_tokens_enabled=1")
            con.commit()
        except sqlite3.OperationalError:
            pass  # older LS without this table
        row = con.execute(
            "select key from authtoken_token order by created limit 1"
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--token", default=None, help="override the DB token")
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()

    token = args.token or db_token()
    if not token:
        raise SystemExit(
            "No API token found. Log in once in the UI (creates a token), or pass --token."
        )

    with open(args.config, encoding="utf-8") as f:
        label_config = f.read()

    body = json.dumps({"label_config": label_config}).encode()
    req = urllib.request.Request(
        f"{args.url}/api/projects/{args.project}",
        data=body,
        method="PATCH",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
            status = resp.status
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise SystemExit(f"PATCH failed: HTTP {e.code}\n{detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach {args.url} — is ./run.sh running? ({e})")

    cfg = data.get("label_config", "")
    print(f"OK: project {args.project} label_config updated (HTTP {status}).")
    print(f"PolygonLabels present: {'PolygonLabels' in cfg}")


if __name__ == "__main__":
    main()
