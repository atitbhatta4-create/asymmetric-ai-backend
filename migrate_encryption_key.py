"""
migrate_encryption_key.py — rotate ENCRYPTION_KEY safely.
Handles ALL exchanges: Bybit, OKX, Binance (all stored in exchange_keys table).

Run from Render shell ONCE after deploy:
  python3 migrate_encryption_key.py

Steps:
  1. Decrypts all stored API keys/secrets/passphrases with current key (or fallback)
  2. Re-encrypts with a freshly generated NEW key
  3. Updates DB
  4. Prints the new key — copy it and set as ENCRYPTION_KEY on Render, then redeploy
"""
import base64, hashlib, os
from cryptography.fernet import Fernet

# ── Old key (whatever the server is currently using) ──────────────────────────
old_raw = os.getenv("ENCRYPTION_KEY", "")
if old_raw:
    old_fernet = Fernet(old_raw.encode())
    print("[migrate] Old key: from ENCRYPTION_KEY env var")
else:
    _fallback = base64.urlsafe_b64encode(hashlib.sha256(b"asym-dev-key").digest())
    old_fernet = Fernet(_fallback)
    print("[migrate] Old key: dev fallback (sha256 of 'asym-dev-key')")

# ── Generate new key ──────────────────────────────────────────────────────────
new_raw = Fernet.generate_key().decode()
new_fernet = Fernet(new_raw.encode())
print("[migrate] New key generated")

# ── Migrate all exchange keys (Bybit + OKX + Binance) ────────────────────────
from database import db_conn

with db_conn() as conn:
    cur = conn.cursor()
    cur.execute("SELECT email, exchange, api_key, api_secret, passphrase FROM exchange_keys")
    rows = cur.fetchall()
    print(f"[migrate] Found {len(rows)} exchange key records")

    migrated = 0
    for row in rows:
        email    = row["email"]
        exchange = row["exchange"]
        try:
            # Decrypt with old key
            plain_key    = old_fernet.decrypt(row["api_key"].encode()).decode()
            plain_secret = old_fernet.decrypt(row["api_secret"].encode()).decode()
            plain_pass   = ""
            if row.get("passphrase"):
                try:
                    plain_pass = old_fernet.decrypt(row["passphrase"].encode()).decode()
                except Exception:
                    plain_pass = ""

            # Re-encrypt with new key
            new_key    = new_fernet.encrypt(plain_key.encode()).decode()
            new_secret = new_fernet.encrypt(plain_secret.encode()).decode()
            new_pass   = new_fernet.encrypt(plain_pass.encode()).decode() if plain_pass else None

            cur.execute(
                "UPDATE exchange_keys SET api_key=%s, api_secret=%s, passphrase=%s WHERE email=%s",
                (new_key, new_secret, new_pass, email)
            )
            print(f"[migrate]   ✓ {email} ({exchange})")
            migrated += 1
        except Exception as e:
            print(f"[migrate]   ✗ {email} ({exchange}) — {e}")

    conn.commit()

print(f"\n[migrate] Done — {migrated}/{len(rows)} keys migrated")
print(f"\n{'='*60}")
print(f"  COPY THIS KEY AND SET IT ON RENDER:")
print(f"  ENCRYPTION_KEY={new_raw}")
print(f"{'='*60}")
print(f"\n  Render → Environment → ENCRYPTION_KEY → paste above → Save → Redeploy")
print(f"  Do NOT share this key with anyone.")
