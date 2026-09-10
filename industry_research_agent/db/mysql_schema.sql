CREATE DATABASE IF NOT EXISTS industry_agent;
USE industry_agent;
CREATE TABLE IF NOT EXISTS runs (
  run_id VARCHAR(64) PRIMARY KEY, session_id VARCHAR(128) NOT NULL,
  user_id VARCHAR(128) NOT NULL DEFAULT 'anonymous', request_hash CHAR(64) NOT NULL,
  request_text TEXT NOT NULL, status VARCHAR(24) NOT NULL, attempt INT NOT NULL DEFAULT 0,
  owner_id VARCHAR(128), lease_expires_at DOUBLE, fencing_token BIGINT NOT NULL DEFAULT 0,
  progress DOUBLE NOT NULL DEFAULT 0, current_node VARCHAR(128), checkpoint_ref VARCHAR(255),
  artifact_ref VARCHAR(512), error_code VARCHAR(128), error_message TEXT,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, created_at DOUBLE NOT NULL,
  started_at DOUBLE, heartbeat_at DOUBLE, finished_at DOUBLE,
  UNIQUE KEY uq_runs_session_hash (session_id, request_hash),
  KEY ix_runs_status_created (status, created_at),
  KEY ix_runs_lease_status (lease_expires_at, status),
  KEY ix_runs_session_created (session_id, created_at)
) ENGINE=InnoDB;
