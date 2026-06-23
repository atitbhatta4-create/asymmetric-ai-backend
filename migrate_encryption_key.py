"""
migrate_encryption_key.py — rotate ENCRYPTION_KEY safely.

Run from Render shell ONCE:
  NEW_KEY=<new_fernet_key> python3 migrate_encryption_key.py

Steps:
  1. Decrypts all stored API keys with old key (fallback or current ENCRYPTION_KEY)
  2. Re-encrypts with NEW_KEY
  3. Updates DB
  4. Prints: set ENCRYPTION_KEY=<NEW_KEY> on Render then redeploy
"""
import base64, hashlib, os, sys
from cryptography.fernet import Fernet

# ── Old key (whatever is currently in use) ────────────────────────────────────
old_raw = os.getenv("ENCRYPTION_KEY", "")
if old_raw:
    old_fernet = Fernet(old_raw.encode())
    print(f"[migrate] Old key: from ENCRYPTION_KEY env var")
else:
    _fallback = base64.urlsafe_b64encode(hashlib.sha256(b"asym-dev-key").digest())
    old_fernet = Fernet(_fallback)
    print(f"[migrate] Old key: dev fallback (sha256 of 'asym-dev-key')")

# ── New key ───────────────────────────────────────────────────────────────────
new_raw = os.getenv("NEW_KEY", "")
if not new_raw:
    new_raw = Fernet.generate_key().decode()
    print(f"[migrate] Generated new key: {new_raw}")
    print(f"[migrate] ⚠️  Copy this key now — set it as ENCRYPTION_KEY on Render after migration")
else:
    print(f"[migrate] New key: from NEW_KEY env var")
new_fernet = Fernet(new_raw.encode())

# ── Migrate ───────────────────────────────────────────────────────────────────
from database import db_conn

with db_conn() as conn:
    cur = conn.cursor()
    cur.execute("SELECT email, bybit_api_key, bybit_api_secret FROM users WHERE bybit_api_key IS NOT NULL AND bybit_api_key != ''")
    rows = cur.fetchall()
    print(f"[migrate] Found {len(rows)} users with API keys")

    migrated = 0
    for row in rows:
        email = row["email"]
        try:
            # Decrypt with old key
            old_key    = old_fernet.decrypt(row["bybit_api_key"].encode()).decode()
            old_secret = old_fernet.decrypt(row["bybit_api_secret"].encode()).decode()
            # Re-encrypt with new key
            new_key    = new_fernet.encrypt(old_key.encode()).decode()
            new_secret = new_fernet.encrypt(old_secret.encode()).decode()
            cur.execute("UPDATE users SET bybit_api_key=%s, bybit_api_secret=%s WHERE email=%s",
                        (new_key, new_secret, email))
            print(f"[migrate]   ✓ {email}")
            migrated += 1
        except Exception as e:
            print(f"[migrate]   ✗ {email} — {e} (skipped)")

    conn.commit()
    print(f"\n[migrate] Done — {migrated}/{len(rows)} keys migrated")
    print(f"\n[migrate] Next step: set ENCRYPTION_KEY={new_raw} in Render env vars → redeploy")
