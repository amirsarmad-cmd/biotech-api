# Home-PC SA scraper

Pulls Seeking Alpha **Premium article bodies** through the user's residential IP + valid Chrome cookies, sending them to the Railway backend's authenticated `/admin/news/ingest` endpoint. Bypasses PerimeterX (which blocks all datacenter IPs including Railway's).

## One-time setup

1. **Install Python deps**:
   ```
   pip install --user requests feedparser beautifulsoup4
   ```

2. **Create config dir**: `%USERPROFILE%\.biotech-news-scraper\`
   - `sa-cookies.json` — export from Chrome via the [Cookie Editor extension](https://chromewebstore.google.com/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm) while logged into seekingalpha.com (⚙ → Export → JSON).
   - `ingest-token.txt` — single-line file containing the value of `NEWS_INGEST_TOKEN` env var on Railway. Already created if you ran the scaffolding script that wrote it from `.claude/news-ingest-token`.
   - `tickers.txt` (optional) — one ticker per line; lines starting with `#` ignored. If absent, falls back to a built-in ~40-ticker biotech list.

3. **Test run** (no Task Scheduler yet):
   ```
   python home_pc_sa_scraper.py --tickers MRNA,PFE --max-bodies 3 --verbose
   ```
   Expected: `bodies: N fetched, 0 blocked` for both tickers, `enriched: 6` from the ingest response.

4. **Install scheduled task** (runs every 6 hours):
   ```
   powershell -ExecutionPolicy Bypass -File install_sa_scraper_task.ps1
   ```

## Cookie expiry

SA's auth cookies (`user_remember_token` / `_sapi_session_id`) typically last ~30 days. When they expire:
1. The `last-run.log` will show `bodies: 0 fetched, N blocked` for all tickers.
2. Re-export from Cookie Editor (same Chrome browser still logged in).
3. Replace `~/.biotech-news-scraper/sa-cookies.json`.

The scheduled task picks up the new file on its next run; no re-install needed.

## Cost / rate considerations

- Residential IP requests: free (your home internet).
- Per run, fetches ~30 XML items + up to `--max-bodies` (default 10) full article bodies per ticker.
- 40 tickers × 10 bodies × 4 runs/day = 1,600 article fetches/day.
- SA tolerates this load comfortably for authenticated users (their normal usage pattern).
- If PerimeterX blocks start appearing, lower `--max-bodies` and/or increase `--body-sleep`.

## Manage the scheduled task

```
Get-ScheduledTask -TaskName BiotechSAScraper | Get-ScheduledTaskInfo
Start-ScheduledTask -TaskName BiotechSAScraper
Unregister-ScheduledTask -TaskName BiotechSAScraper -Confirm:$false
```
