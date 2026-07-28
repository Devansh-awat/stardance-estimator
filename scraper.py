#!/usr/bin/env python3
"""Auto-Embedding Scraper for Stardance (scraper.py).
Scrapes projects from stardance.hackclub.com, saves them into stardance.db (2-table schema),
and automatically embeds new projects on the fly using Hack Club AI.

Usage:
  python3 scraper.py <pid>            # Scrapes a single project ID
  python3 scraper.py discover         # Scan upward from max ID for new projects
  python3 scraper.py refresh [n]      # Refresh due projects, at most n (default 400)
  python3 scraper.py backfill         # One-shot: give NULL rows a refresh window
"""

import os
import sys
import re
import json
import time
import sqlite3
import urllib.request
import urllib.error
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "stardance.db")
ENV_FILE = os.path.join(HERE, ".env")
SITE_URL = "https://stardance.hackclub.com"
EMBED_URL = "https://ai.hackclub.com/proxy/v1/embeddings"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_DIM = 4096          # qwen3-embedding-8b, what embeddings_v2 holds


def _load_key():
    """HACKCLUB_AI_KEY from the environment, else from .env beside this file.

    Never inline it: this file is public. The .env fallback exists because cron
    runs with no environment of its own.
    """
    k = os.environ.get("HACKCLUB_AI_KEY", "").strip()
    if k:
        return k
    try:
        with open(ENV_FILE) as f:
            for line in f:
                if line.startswith("HACKCLUB_AI_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


API_KEY = _load_key()

# How long before a project is looked at again. A shipped project can gain votes
# and a new multiplier, so it moves; a draft rarely does; a 404 is gone for good.
#
# These have to fit inside REFRESH_BATCH x 24. 2,854 shipped daily plus 37,795
# drafts every 3 days is ~15,400 scrapes/day, which an hourly 400 cannot serve --
# the backlog would grow forever. At 7 days the drafts cost ~5,400/day, so the
# whole archive needs ~8,300/day against a 12,000/day ceiling.
SHIPPED_INTERVAL = 86400.0        # 1 day
DRAFT_INTERVAL = 604800.0         # 7 days
DEAD_INTERVAL = 864000.0          # 10 days

# Ceiling on one refresh run. At 0.5s/project this is ~4 minutes of work, so an
# hourly cron stays far inside its tick, and 500/hour is 0.14 req/s at the site.
# See refresh_due_projects.
REFRESH_BATCH = int(os.environ.get("STARDANCE_REFRESH_BATCH", "500"))

# ---------------- DATABASE AUTO-UPGRADE ----------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Add last_checked and refresh_after columns if they don't exist
    try:
        c.execute("ALTER TABLE projects ADD COLUMN last_checked REAL")
        c.execute("ALTER TABLE projects ADD COLUMN refresh_after REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    return conn

# ---------------- HTML PARSING REGEXES ----------------
RE_USER  = re.compile(r'class="project-show__author"[^>]*href="/@([^"]+)"')
RE_TITLE = re.compile(r'<meta property="og:title" content="([^"]*)"')
RE_ART   = re.compile(r'<article\b', re.I)
RE_PTYPE = re.compile(r'data-feed-engagement-post-type-value="Post::(\w+)"')
RE_PID   = re.compile(r'data-feed-engagement-post-id-value="(\d+)"')
RE_TIME  = re.compile(r'<time[^>]*datetime="([^"]+)"')
RE_MULT  = re.compile(r'<span>([\d.]+)x multiplier</span>')
RE_PAY   = re.compile(r'<span>([\d,]+) Stardust</span>')
RE_HOURS = re.compile(r'<span>([\d.]+)h(?:\s*build)?</span>')
RE_DUR   = re.compile(r'feed-post-card__duration">([\d.]+)h')
RE_BLESS = re.compile(r'Blessed \(\+20% bonus\)|title="Blessed')
RE_CURSE = re.compile(r'Cursed \(-50% reduction\)|title="Cursed')
# Kept only as a fallback for markup body_text() cannot parse. These take just
# the FIRST <p> of the body, which truncated 64% of ship writeups -- anything
# with a markdown heading or more than one paragraph lost everything after its
# first line ("# Zero" was the entire stored writeup for a 1,500-char post).
# Use body_text() instead; it walks the whole body subtree.
RE_TEXT  = re.compile(r'project-show__latest-ship-text">\s*<p>(.*?)</p>', re.S)
RE_BODY  = re.compile(r'feed-post-card__body[^>]*>\s*<p>(.*?)</p>', re.S)
RE_DEVLOG_BODY = re.compile(
    r'feed-post-card__body[^>]*markdown-content[^>]*>\s*<p>(.*?)</p>', re.S)
RE_REPO  = re.compile(r'href="([^"]+)">See source code')
RE_DEMO  = re.compile(r'href="([^"]+)">Try project')
RE_TAG   = re.compile(r"<[^>]+>")


class _BodyText(HTMLParser):
    """Collects the full text of every <div> whose class contains `marker`.

    A regex cannot do this: the body holds nested markup (headings, lists, code
    blocks, nested divs), so matching to the first </p> or the first </div> both
    truncate. Track depth instead and take the whole subtree.
    """

    def __init__(self, marker):
        super().__init__(convert_charrefs=True)
        self.marker, self.depth, self.buf, self.out = marker, 0, [], []

    def handle_starttag(self, tag, attrs):
        if self.depth:
            if tag == "div":
                self.depth += 1
            return
        if tag == "div" and self.marker in dict(attrs).get("class", ""):
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth and tag == "div":
            self.depth -= 1
            if self.depth == 0:
                self.out.append(" ".join("".join(self.buf).split()))
                self.buf = []

    def handle_data(self, data):
        if self.depth:
            self.buf.append(data + " ")


def body_text(fragment, marker="feed-post-card__body"):
    """Longest body-div text in `fragment`, or "" if there is none."""
    p = _BodyText(marker)
    try:
        p.feed(fragment)
        p.close()
    except Exception:
        return ""
    texts = [t for t in p.out if t.strip()]
    return max(texts, key=len) if texts else ""


def clean_text(s):
    if not s: return ""
    s = RE_TAG.sub(" ", s)
    s = (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
          .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    return re.sub(r"\s+", " ", s).strip()

def split_articles(html):
    idx = [m.start() for m in RE_ART.finditer(html)]
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(html)
        yield html[s:e]

def parse_project_html(pid, html):
    tit = RE_TITLE.search(html)
    user = RE_USER.search(html)
    
    title = clean_text(tit.group(1)) if tit else ""
    username = user.group(1) if user else ""
    
    ships = []
    devlogs = []
    ship_count = 0
    devlog_hours = 0.0
    ship_hours = 0.0
    total_payout = 0
    
    devlog_texts = []
    ship_texts = []
    
    for a in split_articles(html):
        ptype = RE_PTYPE.search(a)
        if not ptype: continue
        kind = ptype.group(1)
        post_id = int(RE_PID.search(a).group(1)) if RE_PID.search(a) else None
        when = RE_TIME.search(a).group(1) if RE_TIME.search(a) else ""
        
        if kind == "ShipEvent":
            ship_count += 1
            m, p, h = RE_MULT.search(a), RE_PAY.search(a), RE_HOURS.search(a)
            repo, demo = RE_REPO.search(a), RE_DEMO.search(a)

            hours = float(h.group(1)) if h else 0.0
            payout = int(p.group(1).replace(",", "")) if p else 0
            blessing = "blessed" if RE_BLESS.search(a) else ("cursed" if RE_CURSE.search(a) else "neutral")
            text = (body_text(a, "project-show__latest-ship-text")
                    or body_text(a, "feed-post-card__body"))
            if not text:
                txt = RE_TEXT.search(a) or RE_BODY.search(a)
                text = clean_text(txt.group(1)) if txt else ""
            
            ship_hours += hours
            total_payout += payout
            if text: ship_texts.append(text)
            
            ships.append({
                "ship_num": ship_count,
                "ts": when,          # needed to work out which devlogs are unshipped
                "hours": hours,
                "multiplier": float(m.group(1)) if m else None,
                "payout": payout,
                "blessing": blessing,
                "text": text,
                "repo": repo.group(1) if repo else "",
                "demo": demo.group(1) if demo else ""
            })
            
        elif kind == "Devlog":
            d = RE_DUR.search(a)
            hours = float(d.group(1)) if d else 0.0
            text = body_text(a, "markdown-content") or body_text(a, "feed-post-card__body")
            if not text:
                b = RE_DEVLOG_BODY.search(a) or RE_BODY.search(a)
                text = clean_text(b.group(1)) if b else ""
            
            devlog_hours += hours
            if text: devlog_texts.append(text)
            
            devlogs.append({
                "post_id": post_id,
                "ts": when,
                "hours": hours,
                "text": text
            })
            
    # Construct chronological embedding_text
    parts = []
    if title: parts.append(f"Title: {title}")
    if devlog_texts: parts.append("Devlogs:\n" + "\n".join(reversed(devlog_texts)))
    if ship_texts: parts.append("Ships:\n" + "\n".join(reversed(ship_texts)))
    embedding_text = "\n\n".join(parts)
    
    return {
        "pid": pid,
        "title": title,
        "username": username,
        "n_ships": ship_count,
        "total_hours": round(devlog_hours + ship_hours, 2),
        "total_payout": total_payout,
        "ships_json": json.dumps(ships, ensure_ascii=False),
        "devlogs_json": json.dumps(devlogs, ensure_ascii=False),
        "embedding_text": embedding_text
    }

# ---------------- EMBEDDING HELPER ----------------
def generate_embedding(text):
    if not text.strip():
        return None
    if not API_KEY:
        # Scraping still works without a key; the project just lands unembedded.
        print("  no HACKCLUB_AI_KEY (environment or .env) -- skipping embedding")
        return None
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(EMBED_URL, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        return data["data"][0]["embedding"]

# ---------------- MAIN SCRAPER FUNCTION ----------------
def fetch_and_save_project(pid: int):
    url = f"{SITE_URL}/projects/{pid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Stardance Scraper)"})
    
    now = time.time()
    conn = get_db_connection()
    c = conn.cursor()

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # NEVER overwrite a project we already hold. INSERT OR REPLACE here
            # blanked the row, and for a labelled project that destroys the
            # multiplier -- ground truth that cannot be re-derived once the page
            # is gone. Harmless while `discover` was the only caller (its 404s
            # are unscraped pids), but `refresh` walks projects we DO hold, so a
            # deleted page or one bad 404 from the site would silently delete
            # labels. Existing rows only get their window pushed out.
            c.execute("SELECT n_ships FROM projects WHERE pid = ?", (pid,))
            known = c.fetchone()
            if known:
                print(f"[PID {pid}] 404 but already archived ({known['n_ships']} ships) "
                      f"-- keeping the row, backing off {DEAD_INTERVAL/86400:.0f}d")
                c.execute("UPDATE projects SET last_checked = ?, refresh_after = ? "
                          "WHERE pid = ?", (now, now + DEAD_INTERVAL, pid))
            else:
                print(f"[PID {pid}] 404 Not Found")
                c.execute("""
                    INSERT INTO projects
                    (pid, title, username, n_ships, total_hours, total_payout, ships_json, devlogs_json, embedding_text, last_checked, refresh_after)
                    VALUES (?, NULL, NULL, 0, 0.0, 0, '[]', '[]', '', ?, ?)
                """, (pid, now, now + DEAD_INTERVAL))
            conn.commit()
        else:
            print(f"[PID {pid}] HTTP Error {e.code}")
        conn.close()
        return False
    except Exception as e:
        print(f"[PID {pid}] Fetch Error: {e}")
        conn.close()
        return False
        
    proj_data = parse_project_html(pid, html)
    
    # Calculate refresh interval:
    # 1. Active project (shipped): refresh in 1 day
    # 2. Draft / No ships: refresh in 3 days
    if proj_data["n_ships"] > 0:
        refresh_delay = SHIPPED_INTERVAL
    else:
        refresh_delay = DRAFT_INTERVAL

    # Read the old row BEFORE overwriting it: needed both to decide whether the
    # embedding went stale and to refuse a destructive write.
    c.execute("SELECT embedding_text, n_ships, ships_json FROM projects WHERE pid = ?", (pid,))
    prev = c.fetchone()
    text_changed = prev is not None and (prev["embedding_text"] or "") != proj_data["embedding_text"]

    # A project never loses ships. If the parse says zero and we already hold
    # some, the markup changed or the page came back half-rendered -- writing
    # that would erase multipliers, which are the only ground truth there is and
    # are unrecoverable once the ship scrolls out of the feed. Back off instead
    # and leave the row alone; the next tick re-reads it.
    if prev and (prev["n_ships"] or 0) > 0 and proj_data["n_ships"] == 0:
        print(f"[PID {pid}] REFUSED: parsed 0 ships but {prev['n_ships']} are stored. "
              f"Not overwriting. Retrying in 1d.")
        c.execute("UPDATE projects SET last_checked = ?, refresh_after = ? WHERE pid = ?",
                  (now, now + SHIPPED_INTERVAL, pid))
        conn.commit()
        conn.close()
        return False

    # Save project data
    c.execute("""
        INSERT OR REPLACE INTO projects 
        (pid, title, username, n_ships, total_hours, total_payout, ships_json, devlogs_json, embedding_text, last_checked, refresh_after)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        proj_data["pid"], proj_data["title"], proj_data["username"],
        proj_data["n_ships"], proj_data["total_hours"], proj_data["total_payout"],
        proj_data["ships_json"], proj_data["devlogs_json"], proj_data["embedding_text"],
        now, now + refresh_delay
    ))
    conn.commit()
    print(f"[PID {pid}] Saved '{proj_data['title']}' ({proj_data['n_ships']} ships, {proj_data['total_hours']}h)")
    
    # Embed into embeddings_v2 -- the table peer search actually reads. This used
    # to write JSON into the old `embeddings` table, which nothing has read since
    # the v2 migration, so every one of those calls spent the Hack Club daily
    # budget filling a dead table.
    #
    # Only when the vector is missing or its source text changed. A refresh that
    # finds a project unchanged must not re-embed it: the budget is $3/day shared
    # with embedder.py, and most refreshes change nothing.
    c.execute("SELECT pid FROM embeddings_v2 WHERE pid = ?", (pid,))
    needs_vector = (not c.fetchone()) or text_changed
    if needs_vector and proj_data["embedding_text"]:
        why = "text changed" if text_changed else "new"
        print(f"[PID {pid}] Embedding ({why})...")
        try:
            vec = generate_embedding(proj_data["embedding_text"])
            if vec and len(vec) == EMBED_DIM:
                import numpy as np
                c.execute("INSERT OR REPLACE INTO embeddings_v2 (pid, vec) VALUES (?, ?)",
                          (pid, np.asarray(vec, dtype=np.float32).tobytes()))
                conn.commit()
                print(f"[PID {pid}] Embedded vector saved!")
            elif vec:
                # Mixing widths in one table silently produces nonsense
                # neighbours, so refuse rather than corrupt the pool.
                print(f"[PID {pid}] SKIPPED embedding: got {len(vec)} dims, "
                      f"embeddings_v2 holds {EMBED_DIM}")
        except Exception as e:
            print(f"[PID {pid}] Embedding error: {e}")

    conn.close()
    return True

# ---------------- DISCOVERY MODE ----------------
def discover_new_projects():
    """Scans upward from the current highest project ID until 50 consecutive 404s are found."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT MAX(pid) FROM projects")
    max_pid = c.fetchone()[0] or 0
    conn.close()
    
    start_pid = max_pid + 1
    print(f"Starting discovery mode from PID {start_pid}...")
    
    consecutive_404s = 0
    pid = start_pid
    
    while consecutive_404s < 50:
        success = fetch_and_save_project(pid)
        if success:
            consecutive_404s = 0
        else:
            consecutive_404s += 1
        pid += 1
        time.sleep(0.5) # Adaptive rate-limiting friendly gap

    print(f"Discovery mode complete. Checked up to PID {pid - 1}.")

# ---------------- REFRESH MODE ----------------
def backfill_refresh_windows(seed=20260727):
    """Give every project a refresh_after, spread across its own interval.

    `last_checked` and `refresh_after` were added by ALTER TABLE long after the
    archive was built, and SQLite backfills a new column with NULL. 36,049 of
    40,649 rows were therefore NULL, which `refresh_due_projects` read as "due
    now" -- so an hourly cron meant re-scraping the whole archive, hourly, with
    runs overlapping each other. This is the one-shot that repairs it.

    Windows are RANDOMISED across the interval rather than set to now+interval,
    or the entire archive would simply come due together again one day later.
    """
    import random
    rng = random.Random(seed)
    now = time.time()

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT pid, n_ships, last_checked FROM projects "
              "WHERE refresh_after IS NULL")
    rows = c.fetchall()
    print(f"{len(rows):,} projects have no refresh window.")

    updates = []
    for r in rows:
        interval = SHIPPED_INTERVAL if (r["n_ships"] or 0) > 0 else DRAFT_INTERVAL
        checked = r["last_checked"] or now
        updates.append((checked, now + rng.uniform(0, interval), r["pid"]))

    c.executemany("UPDATE projects SET last_checked = ?, refresh_after = ? WHERE pid = ?",
                  updates)
    conn.commit()
    left = c.execute("SELECT COUNT(*) FROM projects WHERE refresh_after IS NULL").fetchone()[0]
    conn.close()
    print(f"Backfilled {len(updates):,}. Still NULL: {left}.")
    return len(updates)


def refresh_due_projects(limit=REFRESH_BATCH):
    """Re-scrape the projects whose refresh window has passed, oldest first.

    HARD CAP on purpose. A refresh run is bounded by `limit` no matter what the
    query returns, so a bad window (or a fresh NULL from some future migration)
    can never again turn one cron tick into a full-archive re-scrape. Anything
    left over is simply picked up by the next tick, because it stays overdue.
    """
    conn = get_db_connection()
    c = conn.cursor()
    now = time.time()
    # NULLs sort first in SQLite ASC, so a missed backfill surfaces here rather
    # than hiding behind 36k rows -- and the cap keeps it survivable either way.
    c.execute("SELECT pid FROM projects WHERE refresh_after IS NULL OR refresh_after <= ? "
              "ORDER BY refresh_after ASC LIMIT ?", (now, limit))
    due_pids = [r["pid"] for r in c.fetchall()]
    total_due = c.execute(
        "SELECT COUNT(*) FROM projects WHERE refresh_after IS NULL OR refresh_after <= ?",
        (now,)).fetchone()[0]
    conn.close()

    print(f"{total_due:,} projects due; refreshing {len(due_pids):,} this run "
          f"(cap {limit}).")

    for idx, pid in enumerate(due_pids, 1):
        print(f"[{idx}/{len(due_pids)}] Refreshing project {pid}...")
        fetch_and_save_project(pid)
        time.sleep(0.5)

    print("Refresh mode complete.")

# ---------------- CLI ENTRYPOINT ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 scraper.py <pid>          # Scrape single project")
        print("  python3 scraper.py discover       # Discover new projects")
        print("  python3 scraper.py refresh [n]    # Refresh due projects (cap n)")
        print("  python3 scraper.py backfill       # Repair NULL refresh windows")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "discover":
        discover_new_projects()
    elif cmd == "refresh":
        refresh_due_projects(int(sys.argv[2]) if len(sys.argv) > 2 else REFRESH_BATCH)
    elif cmd == "backfill":
        backfill_refresh_windows()
    else:
        try:
            target_pid = int(cmd)
            fetch_and_save_project(target_pid)
        except ValueError:
            print(f"Unknown command or PID: {cmd}")
