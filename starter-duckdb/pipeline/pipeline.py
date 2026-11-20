#!/usr/bin/env python3
"""Run the Shopping Sessions pipeline and write the two scored Parquet tables."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import duckdb

from input_csv import CAMPAIGNS, CLICKS, EVENTS, PRODUCTS, prepare


WEEK_MS = 604_800_000

# Original starter checklist, kept as a guide. The working code below
# implements these steps.
#
# TODO: implement the rules.
#   Events: parse -> dedup on event_id (smallest ingest_ts, before validation)
#           -> validate -> drop late -> sessionize per user (gap > 30 min)
#           -> day of session start -> categories
#   Clicks: parse -> dedup -> validate -> credit each session to the latest
#           click at or before its start, within 7 days (ties: smallest click_id)
#
# Careful with DuckDB's CSV reader: it repairs or rejects malformed lines in its
# own way, and it reads a quoted empty field "" as '' rather than null. Check its
# behaviour against the fixtures before trusting it with the dialect rules.


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def register_csv(con, name, path, columns):
    schema = ", ".join(f"'{column}': 'VARCHAR'" for column in columns)
    con.execute(f"""
        CREATE VIEW {name} AS
        SELECT * FROM read_csv({literal(path)}, header=true, auto_detect=false,
                               delim=',', new_line='\\n', nullstr='', columns={{{schema}}})
    """)


def export(con, output, name, query):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    path = literal(directory / "part-0.parquet")
    con.execute(f"COPY ({query}) TO {path} (FORMAT PARQUET, COMPRESSION SNAPPY)")


def report(stage, started):
    print(f"{stage} in {perf_counter() - started:.2f}s", file=sys.stderr)


def run(con, output, scratch, window):
    con.execute("""
        CREATE MACRO iso_ms(s) AS
            epoch_ms(TRY_STRPTIME(s, '%Y-%m-%dT%H:%M:%S'));
        CREATE MACRO timestamp_ms(s) AS CASE
            WHEN regexp_full_match(s, '[0-9]+')
                THEN TRY_CAST(COALESCE(NULLIF(ltrim(s, '0'), ''), '0') AS BIGINT)
            WHEN regexp_full_match(s,
                    '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z')
                THEN iso_ms(left(s, 19))
            WHEN regexp_full_match(s,
                    '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-]([01][0-9]|2[0-3]):[0-5][0-9]')
                THEN iso_ms(left(s, 19)) - CASE WHEN s[20] = '-' THEN -1 ELSE 1 END
                    * (CAST(s[21:22] AS BIGINT) * 60 + CAST(s[24:25] AS BIGINT)) * 60000
            END;
    """)
    start, end = con.execute(
        "SELECT timestamp_ms(?), timestamp_ms(?)",
        [window["window_start"], window["window_end"]],
    ).fetchone()

    register_csv(con, "raw_events", scratch / "events-*.csv", EVENTS)
    register_csv(con, "raw_clicks", scratch / "clicks-*.csv", CLICKS)
    register_csv(con, "identities", scratch / "identity.csv", ("source_id", "person_id"))
    register_csv(con, "products", scratch / "products.csv", PRODUCTS)
    register_csv(con, "campaigns", scratch / "campaigns.csv", CAMPAIGNS)

    # Clean events, group them into visits, and write one row per person and day.
    started = perf_counter()
    # Treat a winning duplicate as one complete row, including null fields.
    con.execute("""
        CREATE TEMP TABLE session_events AS
        WITH dedup AS (
            SELECT arg_min(
                struct_pack(user_id := user_id, ts := ts, ingest_ts := ingest_ts,
                            event_type := event_type, product_id := product_id, price_cents := price_cents),
                ingest_ts
            ) AS event
            FROM raw_events
            GROUP BY event_id
        ), decoded AS (
            SELECT event.user_id AS user_id, timestamp_ms(event.ts) AS ts,
                   iso_ms(left(event.ingest_ts, 19)) AS ingest,
                   event.product_id AS product_id, event.event_type AS event_type,
                   CASE WHEN event.event_type = 'purchase' AND regexp_full_match(event.price_cents, '[0-9]+')
                        THEN TRY_CAST(COALESCE(NULLIF(ltrim(event.price_cents, '0'), ''), '0') AS INTEGER)
                   END AS cents
            FROM dedup
        ), valid AS (
            SELECT coalesce(person_id, user_id) AS user_id, ts,
                   CASE WHEN event_type = 'purchase' THEN product_id END AS purchased_product,
                   coalesce(cents, 0) AS cents
            FROM decoded
            LEFT JOIN identities ON source_id = user_id
            WHERE user_id IS NOT NULL AND product_id IS NOT NULL
              AND ts >= ? AND ts < ? AND ingest - ts <= 86400000
              AND (event_type IN ('view', 'cart') OR (event_type = 'purchase' AND cents BETWEEN 1 AND 10000000))
        ), with_gaps AS (
            SELECT *, ts - lag(ts) OVER (PARTITION BY user_id ORDER BY ts) AS gap
            FROM valid
        )
        SELECT user_id, purchased_product, cents,
               max(CASE WHEN gap IS NULL OR gap > 1800000 THEN ts END) OVER (
                   PARTITION BY user_id ORDER BY ts
                   RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS session_start
        FROM with_gaps
    """, [start, end])
    con.execute("""
        CREATE TEMP TABLE sessions AS
        SELECT user_id, session_start, CAST(make_timestamp_ms(session_start) AS DATE) AS day,
               count(*) AS events, sum(cents)::BIGINT AS revenue_cents
        FROM session_events
        GROUP BY user_id, session_start;
    """)
    export(con, output, "user_daily", """
        WITH daily AS (
            SELECT user_id, day, count(*) AS sessions, sum(events)::BIGINT AS events,
                   sum(revenue_cents)::BIGINT AS revenue_cents
            FROM sessions GROUP BY user_id, day
        ), category_totals AS (
            SELECT user_id, CAST(make_timestamp_ms(session_start) AS DATE) AS day,
                   coalesce(category, 'unknown') AS category, sum(cents) AS revenue
            FROM session_events
            LEFT JOIN products ON purchased_product = product_id
            WHERE purchased_product IS NOT NULL
            GROUP BY user_id, day, coalesce(category, 'unknown')
        ), top_categories AS (
            SELECT user_id, day,
                   min(struct_pack(negative_revenue := -revenue, name := category)).name AS top_category
            FROM category_totals GROUP BY user_id, day
        )
        SELECT user_id, day, sessions, events, revenue_cents, top_category
        FROM daily LEFT JOIN top_categories USING (user_id, day)
    """)
    con.execute("DROP TABLE session_events")
    report("Built sessions and wrote user_daily", started)

    # Clean clicks and map each click to the same person IDs as the events.
    started = perf_counter()
    con.execute("""
        CREATE TEMP TABLE clicks AS
        WITH valid AS (
            SELECT click_id, user_id, campaign_id, timestamp_ms(ts) AS ts
            FROM raw_clicks
            WHERE user_id IS NOT NULL
        )
        SELECT coalesce(person_id, user_id) AS user_id, valid.campaign_id, valid.ts,
               min(click_id) AS click_id
        FROM valid
        INNER JOIN campaigns USING (campaign_id)
        LEFT JOIN identities ON source_id = user_id
        WHERE ts >= ? AND ts < ?
        GROUP BY coalesce(person_id, user_id), valid.campaign_id, valid.ts
    """, [start - WEEK_MS, end])
    report("Prepared valid clicks", started)

    # Credit each session once and count each campaign that helped bring it in.
    started = perf_counter()
    export(con, output, "campaign_daily", f"""
        WITH best_clicks AS (
            SELECT user_id, ts, arg_min(campaign_id, click_id) AS campaign_id
            FROM clicks GROUP BY user_id, ts
        ), credited AS (
            SELECT CASE WHEN c.ts >= s.session_start - {WEEK_MS} THEN c.campaign_id
                        ELSE 'organic' END AS campaign_id,
                   s.day, count(*) AS sessions,
                   count(*) FILTER (WHERE revenue_cents > 0) AS purchasing_sessions,
                   sum(revenue_cents)::BIGINT AS revenue_cents
            FROM sessions s
            ASOF LEFT JOIN best_clicks c ON s.user_id = c.user_id AND s.session_start >= c.ts
            GROUP BY 1, 2
        ), person_days AS (
            SELECT user_id, day, min(session_start) AS first_start, max(session_start) AS last_start
            FROM sessions GROUP BY user_id, day
        ), intervals AS (
            -- The next click for this campaign covers any overlap. Keeping
            -- these intervals disjoint prevents duplicate assists.
            SELECT user_id, campaign_id, ts AS lo,
                   least(ts + {WEEK_MS},
                         lead(ts) OVER (PARTITION BY user_id, campaign_id ORDER BY ts) - 1) AS hi
            FROM clicks
        ), endpoints AS (
            SELECT i.user_id, i.campaign_id, d.day, boundary.ts, boundary.sign
            FROM intervals i
            JOIN person_days d ON i.user_id = d.user_id AND i.lo <= d.last_start AND i.hi >= d.first_start
            CROSS JOIN LATERAL (VALUES (i.hi, 1), (i.lo - 1, -1)) boundary(ts, sign)
        ), ranks AS (
            SELECT user_id, session_start, day,
                   row_number() OVER (PARTITION BY user_id, day ORDER BY session_start) AS session_rank
            FROM sessions
        ), assisted AS (
            -- Subtract the rank at lo - 1 from the rank at hi to count [lo, hi].
            SELECT e.campaign_id, e.day,
                   sum(coalesce(r.session_rank, 0) * e.sign)::BIGINT AS assisted_sessions
            FROM endpoints e
            ASOF LEFT JOIN ranks r ON e.user_id = r.user_id AND e.day = r.day AND e.ts >= r.session_start
            GROUP BY 1, 2
            HAVING assisted_sessions > 0
        )
        SELECT campaign_id, coalesce(channel, 'organic') AS channel, day,
               coalesce(sessions, 0) AS sessions,
               coalesce(purchasing_sessions, 0) AS purchasing_sessions,
               coalesce(revenue_cents, 0) AS revenue_cents,
               coalesce(assisted_sessions, 0) AS assisted_sessions
        FROM credited FULL JOIN assisted USING (campaign_id, day)
        LEFT JOIN campaigns USING (campaign_id)
    """)
    report("Attributed sessions and wrote campaign_daily", started)


def main():
    data = Path(os.environ.get("DATA_DIR", "/data"))
    output = Path(os.environ.get("OUTPUT_DIR", "/output"))
    scratch = Path(os.environ.get("SCRATCH_DIR", "/scratch"))
    threads = int(os.environ.get("NCPU", str(min(8, os.cpu_count() or 1))))
    memory = int(os.environ.get("MEM_GB", "8"))
    window = json.loads((data / "manifest.json").read_text())
    started = perf_counter()

    with TemporaryDirectory(dir=scratch) as directory:
        work = Path(directory)
        prepare(data, work, threads)
        report("Prepared input", started)
        with duckdb.connect() as con:
            con.execute(f"SET threads = {threads}")
            con.execute(f"SET memory_limit = '{max(1, int(memory * 0.75))}GB'")
            con.execute(f"SET temp_directory = {literal(work / 'duckdb_tmp')}")
            con.execute("SET preserve_insertion_order = false")
            run(con, output, work, window)

    report("Completed pipeline", started)


if __name__ == "__main__":
    main()
