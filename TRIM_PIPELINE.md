# Email trimming pipeline (SLM summariser + live ChatGPT product)

Turns raw incoming email (Outlook/Word HTML, deep quoted chains, forwards,
multilingual boilerplate) into the minimal genuine message. Serves two
consumers: the live ChatGPT summariser and the from-scratch SLM's training
data. Design notes and rationale live in the chat/memory; this is the
operational spec.

Measured effect: a 4MB Word-doc-as-email thread → ~85 chars of real content;
median trimmed email ~568 chars; 93% under the 950-char cap.

## Pre-processing
1. **HTML → text** — strip `<style>`, `<head>`, `<script>` blocks wholesale,
   then drop tags. (Inline images are already extracted to attachments upstream.)
2. **URL-strip:** `https?://\S{30,}` — removes safelinks/tracking URLs. Run this
   **before** block hashing, so signatures that differ only by an embedded
   per-message safelink still dedup consistently.
3. **Split into blocks by newline.**
4. **Normalise each block for hashing:** lowercase, collapse whitespace runs to a
   single space, strip leading `>` / punctuation, trim. (Hash the normalised form;
   keep the original for output.)

## Boilerplate dedup (two-pass, frozen table)
5. **Pass 1** — build a global digest table: MD5 of each normalised block, with
   **frequency = number of distinct emails** it appears in. Freeze the table after
   a full pass over the corpus.
6. **Pass 2 (read-only against the frozen table)** — drop a block only if
   **frequency ≥ 10 (tunable via config) AND length ≥ 200 chars**. Both conditions
   are required:
   - length floor protects short legitimate content that happens to recur
     ("Please find the attached report");
   - frequency floor separates boilerplate (hundreds of occurrences) from
     coincidentally-shared content (a handful).
7. **Skip-list** for known template noise lines — fast/deterministic; frequency
   would also catch them but this is cheaper and exact:
   - `=== reply above this line ===` (Zendesk/Freshdesk; content is BELOW it — bin
     the line, do not cut there)
   - `You don't often get email from <addr>. Learn why this is important`
   - `CAUTION`/`EXTERNAL`/`ALERT` external-sender banners
   - `thread::<token>::` ids

## Included-email / thread-depth cut
8. Detect a quoted-reply header with the **compound match**, NOT bare `^From:`:
   `From:` followed within a line or two by `Sent:`/`Date:` and `To:`. Bare
   `^From:` false-fires on real content ("update the From: address on the invoice").
   Treat **forward separators** as equivalent headers:
   `---------- Forwarded message ----------`, `Begin forwarded message:`
   — and skip leading blank lines before the test (Gmail/Apple forwards don't lead
   with `From:`).
9. **Normal case:** cut at the **first** header — everything below is quoted
   history; drop it.
10. **Pure-forward / trivial-note exception:** if the first substantive line **is**
    a forward header/separator, **OR** the text before the first header is
    **< 50 chars** (tunable; e.g. "FYI") → cut at the **second** header instead, so
    the forwarded body (and any short covering note) is kept. If there is **no**
    second header → **keep to end** (must be "keep to end", not "keep nothing").
11. **Sign-off cut:** cut at the first line that is a sign-off, **allowing a
    trailing name** (this corpus writes `Regards, <Name>` inline, so an
    alone-on-a-line rule almost never fires). Anchor to line start:
    `regards | best regards | kind regards | thanks & regards | thanks and regards |
     thank you | thanks | br | cheers | met vriendelijke groet`
    Bin everything below. This is how multi-line signatures get removed — the
    length floor in step 6 deliberately will not touch them.
12. In practice steps 9–11 combine into one top-down pass: **cut at the earliest of
    {sign-off line, quoted-header block}** (with the forward exception from step 10).

## Cap + filter
13. **Head-cap at 950 chars**, at a **block boundary** (stop appending once the next
    block would take the total over 950). **Always include the first block; if it
    alone exceeds 950, hard-truncate it to 950** — otherwise a single-giant-block
    email emits empty. (950 chars ≈ 448-token `email_budget` for the SLM.)
14. **Pair-level drops at the train/val split step:**
    - zero-block / genuinely-empty emails (attachment-only; e.g. ids 9307, 9405),
      and anything that trims to empty — drop the email **and** its summary together
      (never keep a summary whose email is empty → that trains hallucination);
    - near-duplicate email dedup — keep one representative.

## Two consumers, two final caps
15. **Live ChatGPT product:** 30,000-char backstop (the API input limit; only bites
    pathological outliers).
16. **SLM training regen:** cap the teacher's input at `email_budget` (~950 chars),
    the SAME text the model will see — train/target consistency, and it slashes the
    per-summary input-token cost.

## Config knobs
- `freq_threshold` = 10 (block-drop frequency floor)
- `min_block_len` = 200 (block-drop length floor)
- `short_note_len` = 50 (forward trivial-note threshold)
- `head_cap_chars` = 950 (SLM path) / 30000 (ChatGPT path)
- table rebuild cadence: scheduled in the weekend/low-traffic window; atomic
  temp-file + rename swap; version/date-stamp the table.

## Downstream budget (SLM)
- `email_budget` = 448 tokens (bounded by the 950-char cap; covers the full capped
  output so the packer never clips content the teacher saw)
- `summary_budget` ≈ 128 (PLACEHOLDER — re-measure from the regenerated
  English-forced summaries)
- `S` = email_budget + summary_budget + 2 ≈ 578
