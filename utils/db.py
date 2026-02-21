"""
EcoVision – SQLite Database Layer
===================================
Manages the persistent  notifications  table used by the Anomaly Detector.

Schema
------
notifications
    id          INTEGER  PRIMARY KEY AUTOINCREMENT
    timestamp   TEXT     ISO-8601 datetime of the anomalous reading
    resource    TEXT     'energy' | 'water' | 'waste'
    building    TEXT     Campus location / zone (nullable)
    metric      TEXT     Column name that was anomalous (e.g. 'energy_kwh')
    actual      REAL     Observed value
    predicted   REAL     Model-predicted value
    residual    REAL     actual - predicted
    sigma       REAL     |residual| / rolling_std  (how many σ away)
    severity    TEXT     'critical' (>3σ) | 'high' (>2σ) | 'medium' (1.5–2σ)
    alert_type  TEXT     'phantom_load' | 'unusual_drop' | 'water_spike' |
                         'pressure_drop' | 'waste_spike' | 'waste_drop'
    status      TEXT     'active' | 'acknowledged' | 'resolved'
    resolved_at TEXT     ISO-8601 datetime when resolved (nullable)
    created_at  TEXT     ISO-8601 datetime this row was inserted
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

# Default DB path – sits inside  data/  next to the project root
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ecovision.db'
)


# ─────────────────────────────────────────────────────────────
#  Connection helper
# ─────────────────────────────────────────────────────────────

@contextmanager
def get_conn(db_path: str = DEFAULT_DB_PATH):
    """Context-managed SQLite connection with WAL mode for concurrent reads."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
#  Schema initialisation
# ─────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS notifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    resource    TEXT    NOT NULL,
    building    TEXT,
    metric      TEXT    NOT NULL,
    actual      REAL    NOT NULL,
    predicted   REAL    NOT NULL,
    residual    REAL    NOT NULL,
    sigma       REAL    NOT NULL,
    severity    TEXT    NOT NULL,
    alert_type  TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'active',
    resolved_at TEXT,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notifications_status
    ON notifications (status);

CREATE INDEX IF NOT EXISTS idx_notifications_resource
    ON notifications (resource, timestamp);
"""


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create tables if they do not exist."""
    with get_conn(db_path) as conn:
        conn.executescript(DDL)
    print(f"[DB] Initialised  →  {db_path}")


# ─────────────────────────────────────────────────────────────
#  Write helpers
# ─────────────────────────────────────────────────────────────

def insert_alert(alert: dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Insert a single anomaly alert row.

    Required keys in  alert :
        timestamp, resource, metric, actual, predicted,
        residual, sigma, severity, alert_type
    Optional:
        building, status  (defaults to 'active')

    Returns the new row id.
    """
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    sql = """
        INSERT INTO notifications
            (timestamp, resource, building, metric,
             actual, predicted, residual, sigma,
             severity, alert_type, status, created_at)
        VALUES
            (:timestamp, :resource, :building, :metric,
             :actual, :predicted, :residual, :sigma,
             :severity, :alert_type, :status, :created_at)
    """
    row = {
        'timestamp':  alert['timestamp'],
        'resource':   alert['resource'],
        'building':   alert.get('building'),
        'metric':     alert['metric'],
        'actual':     float(alert['actual']),
        'predicted':  float(alert['predicted']),
        'residual':   float(alert['residual']),
        'sigma':      float(alert['sigma']),
        'severity':   alert['severity'],
        'alert_type': alert['alert_type'],
        'status':     alert.get('status', 'active'),
        'created_at': now,
    }
    with get_conn(db_path) as conn:
        cur = conn.execute(sql, row)
        return cur.lastrowid


def bulk_insert_alerts(alerts: list, db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert many alert dicts in one transaction. Returns count inserted."""
    if not alerts:
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    sql = """
        INSERT INTO notifications
            (timestamp, resource, building, metric,
             actual, predicted, residual, sigma,
             severity, alert_type, status, created_at)
        VALUES
            (:timestamp, :resource, :building, :metric,
             :actual, :predicted, :residual, :sigma,
             :severity, :alert_type, :status, :created_at)
    """
    rows = [
        {
            'timestamp':  a['timestamp'],
            'resource':   a['resource'],
            'building':   a.get('building'),
            'metric':     a['metric'],
            'actual':     float(a['actual']),
            'predicted':  float(a['predicted']),
            'residual':   float(a['residual']),
            'sigma':      float(a['sigma']),
            'severity':   a['severity'],
            'alert_type': a['alert_type'],
            'status':     a.get('status', 'active'),
            'created_at': now,
        }
        for a in alerts
    ]
    with get_conn(db_path) as conn:
        conn.executemany(sql, rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────
#  Read helpers
# ─────────────────────────────────────────────────────────────

def get_active_alerts(
    limit:    int  = 50,
    resource: str  = None,
    db_path:  str  = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Return active alerts ordered by severity then sigma descending.
    Optionally filter by resource ('energy' | 'water' | 'waste').
    """
    sql = """
        SELECT * FROM notifications
        WHERE  status = 'active'
        {resource_filter}
        ORDER BY
            CASE severity
                WHEN 'critical' THEN 1
                WHEN 'high'     THEN 2
                WHEN 'medium'   THEN 3
                ELSE 4
            END,
            sigma DESC
        LIMIT :limit
    """
    params: dict = {'limit': limit}
    if resource:
        sql = sql.format(resource_filter="AND resource = :resource")
        params['resource'] = resource
    else:
        sql = sql.format(resource_filter="")

    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_alert_counts(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Return count of active alerts grouped by severity."""
    sql = """
        SELECT severity, COUNT(*) as cnt
        FROM   notifications
        WHERE  status = 'active'
        GROUP  BY severity
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return {r['severity']: r['cnt'] for r in rows}


def get_recent_alerts(
    hours:   int = 24,
    limit:   int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Return alerts created within the last N hours."""
    sql = """
        SELECT * FROM notifications
        WHERE  created_at >= datetime('now', :offset)
        ORDER  BY created_at DESC
        LIMIT  :limit
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, {'offset': f'-{hours} hours', 'limit': limit}).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
#  Update helpers
# ─────────────────────────────────────────────────────────────

def resolve_alert(alert_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Mark a single alert as resolved."""
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    sql = """
        UPDATE notifications
        SET    status = 'resolved', resolved_at = :now
        WHERE  id = :id AND status != 'resolved'
    """
    with get_conn(db_path) as conn:
        cur = conn.execute(sql, {'now': now, 'id': alert_id})
    return cur.rowcount > 0


def acknowledge_alert(alert_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Mark a single alert as acknowledged (seen but not fixed)."""
    sql = """
        UPDATE notifications
        SET    status = 'acknowledged'
        WHERE  id = :id AND status = 'active'
    """
    with get_conn(db_path) as conn:
        cur = conn.execute(sql, {'id': alert_id})
    return cur.rowcount > 0


def clear_old_alerts(days: int = 30, db_path: str = DEFAULT_DB_PATH) -> int:
    """Hard-delete resolved/acknowledged alerts older than N days."""
    sql = """
        DELETE FROM notifications
        WHERE  status IN ('resolved', 'acknowledged')
        AND    created_at < datetime('now', :offset)
    """
    with get_conn(db_path) as conn:
        cur = conn.execute(sql, {'offset': f'-{days} days'})
    return cur.rowcount
