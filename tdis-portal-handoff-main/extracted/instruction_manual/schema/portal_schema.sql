-- TDIS Portal schema (Source A) — consolidated from tdis-portal-db @ development
-- Excludes seed data, sample data, and truncate scripts.
-- Database: tdis_portal | Schema: public
-- Generated: 2026-08-28

-- ===== 001_enable_extensions.sql =====
-- Enable PostGIS extension for spatial data types and functions
CREATE EXTENSION IF NOT EXISTS postgis;

-- Enable pgcrypto extension to allow UUID generation with gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ===== 002_create_event_types_table.sql =====
-- Create the event_types table to store different types of events that can be detected
CREATE TABLE IF NOT EXISTS event_types (
    event_type_id          SERIAL PRIMARY KEY,                  -- Unique identifier for each event type, auto-incrementing
    event_type             VARCHAR(30) NOT NULL UNIQUE,         -- Short name/code for the event type (e.g., 'Fire', 'Flood', 'Tornado')
    event_type_description TEXT                                 -- Description of the event type(Optional)
);
-- ===== 003_create_data_sources_table.sql =====
-- Create the data_sources table to store information about external data sources
-- This table maintains metadata about various data sources used for event detection
CREATE TABLE IF NOT EXISTS data_sources (
    data_source_id          SERIAL PRIMARY KEY,                  -- Unique identifier for each data source, auto-incrementing
    data_source             VARCHAR(100) UNIQUE NOT NULL,        -- Name or identifier of the data source service
    data_source_description TEXT,                                -- Detailed description of the data source and its capabilities
    api_base_url            VARCHAR(100),                        -- Base URL endpoint for accessing the data source's API
    api_version             VARCHAR(10),                         -- Version number of the API being used (e.g., 'v1', '2.0')
    api_type                VARCHAR(30),                         -- Type of API interface (e.g., 'REST', 'GraphQL', 'SOAP')
    is_api_active           BOOLEAN                              -- Current operational status of the data source (true = active, false = inactive)
);
-- ===== 004_create_counties_table.sql =====
-- Create the counties table to store geographical boundaries of counties
CREATE TABLE IF NOT EXISTS counties (
    county_id   SERIAL PRIMARY KEY,                   -- Unique identifier for each county, auto-incrementing
    county      VARCHAR(50) NOT NULL UNIQUE,          -- Name of the county
    geom        GEOMETRY(MultiPolygon, 4326) NOT NULL -- Spatial geometry of the county boundary in WGS84 (EPSG:4326)
);

-- Add spatial index for faster geospatial queries
-- CREATE INDEX IF NOT EXISTS idx_counties_geom
-- ON counties
-- USING GIST (geom);
-- ===== 005_create_batch_jobs_table.sql =====
-- Create ENUM type for batch job status to ensure consistent status values
-- This enum defines the possible states a batch job can be in
CREATE TYPE batch_job_status AS ENUM (
    'COMPLETED',    -- Job has finished successfully
    'FAILED',       -- Job encountered an error and did not complete
    'STARTED'       -- Job is currently running
);

-- Create the batch_jobs table to track the execution of batch processing jobs

CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_job_id            SERIAL PRIMARY KEY,                  -- Unique identifier for batch job, auto-incrementing
    batch_job               VARCHAR(100) UNIQUE NOT NULL,        -- Descriptive name or identifier of the batch job
    lastrun_timestamp_start TIMESTAMP,                           -- Timestamp when the batch job started execution
    lastrun_timestamp_end   TIMESTAMP,                           -- Timestamp when the batch job completed or failed
    lastrun_duration        INTERVAL,                            -- Time taken to complete the batch job (end - start)
    lastrun_status          batch_job_status                     -- Current status of the batch job (COMPLETED, FAILED, STARTED)
);
-- ===== 006_create_events_store_table.sql =====
-- Create the events_store table to store detected events from various data sources
-- This is the main table that stores all event data with their spatial information
CREATE TABLE IF NOT EXISTS events_store (
    event_store_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(), -- Unique identifier for each event record
    event_type_id       INTEGER NOT NULL,                           -- Reference to the type of event (e.g., Fire, Flood)
    data_source_id      INTEGER NOT NULL,                           -- Reference to the data source that detected the event
    created_at          TIMESTAMP NOT NULL,                         -- Timestamp when the event was detected/recorded
    geom                GEOMETRY(Point, 4326) NOT NULL,             -- Geographic location of the event in WGS84 (EPSG:4326)
    county_id           INTEGER NOT NULL,                           -- Reference to the county where the event occurred
    event_date          DATE NOT NULL,                              -- Date when the event occurred (without time component)
    batch_job_id        INTEGER NOT NULL,                           -- Reference to the batch job that processed this event
    
    -- Ensure no duplicate events of the same type at the same location on the same date
    CONSTRAINT unique_event_type_id_geom_date UNIQUE (event_type_id, geom, event_date),

    -- Foreign key constraints to maintain referential integrity
    FOREIGN KEY (event_type_id) REFERENCES event_types(event_type_id),
    FOREIGN KEY (data_source_id) REFERENCES data_sources(data_source_id),
    FOREIGN KEY (county_id) REFERENCES counties(county_id),
    FOREIGN KEY (batch_job_id) REFERENCES batch_jobs(batch_job_id)
);

-- Add index for created_at to speed up time-based queries, if it doesn't exist
-- CREATE INDEX IF NOT EXISTS idx_created_at ON events_store(created_at);

-- Add index for event_date to optimize date range filtering
CREATE INDEX IF NOT EXISTS idx_event_date ON events_store(event_date);

-- Add spatial index for geom to optimize geospatial queries, if it doesn't exist
-- CREATE INDEX IF NOT EXISTS idx_geom ON events_store USING GIST (geom);


-- ===== 011_view_events_store.sql =====
-- Create view_events_store view to provide a simplified interface for querying events
-- This view joins the events_store table with related tables to provide human-readable information
CREATE VIEW view_events_store AS
SELECT 
    es.event_store_id,    -- Unique identifier of the event
    et.event_type,        -- Type of event (e.g., Fire, Flood)
    ds.data_source,       -- Source that detected the event
    es.geom,              -- Geographic location of the event
    c.county,             -- County where the event occurred
    es.event_date         -- Date when the event occurred
FROM events_store es 
INNER JOIN event_types et 
    ON es.event_type_id = et.event_type_id
INNER JOIN counties c
    ON es.county_id = c.county_id
INNER JOIN data_sources ds
    ON es.data_source_id = ds.data_source_id
INNER JOIN batch_jobs bj
    ON es.batch_job_id = bj.batch_job_id;
-- ===== 013_add_columns_to_events_store.sql =====
-- Add is_active and description columns to events_store table
ALTER TABLE events_store
ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT true, -- Event active status
ADD COLUMN description TEXT; -- Event description 
-- ===== 014_update_view_events_store.sql =====
-- Update view_events_store to include new columns from events_store table (is_active and description)
CREATE OR REPLACE VIEW view_events_store AS
SELECT 
    es.event_store_id,    -- Unique identifier of the event
    et.event_type,        -- Type of event (e.g., Fire, Flood)
    ds.data_source,       -- Source that detected the event
    es.geom,              -- Geographic location of the event
    c.county,             -- County where the event occurred
    es.event_date,        -- Date when the event occurred
    es.is_active,         -- Event active status
    es.description        -- Event description
FROM events_store es 
INNER JOIN event_types et 
    ON es.event_type_id = et.event_type_id
INNER JOIN counties c
    ON es.county_id = c.county_id
INNER JOIN data_sources ds
    ON es.data_source_id = ds.data_source_id
INNER JOIN batch_jobs bj
    ON es.batch_job_id = bj.batch_job_id; 
-- ===== 018_update_event_types_table_data.sql =====
UPDATE event_types
SET event_type = 'High Wind'
WHERE event_type = 'Highwinds'; 
-- ===== 022_cleanup_sample_events.sql =====
-- Description: Removes all sample test data events that were inserted for testing purposes
-- Impact: This migration will permanently delete all events with descriptions containing "(Test data)"

-- Delete only the sample test data events we inserted
DELETE FROM events_store
WHERE description LIKE '%(Test data)%'; 
-- ===== 023_add_batch_jobs_index.sql =====
/*
 * Migration: Add index to batch_jobs table
 * Description: Creates an index on batch_job and lastrun_timestamp_end columns
 * to optimize the batch_jobs_latest view query performance.
 */
CREATE INDEX IF NOT EXISTS idx_batch_jobs_job_timestamp 
ON batch_jobs (batch_job, lastrun_timestamp_end DESC); 
-- ===== 024_view_batch_jobs_latest.sql =====
/*
 * View: view_batch_jobs_latest
 * Description: This view provides the most recent execution timestamp for each batch job.
 * It aggregates the batch_jobs table to show only the latest run timestamp (lastrun_timestamp_end)
 * for each unique batch_job.
 */
CREATE OR REPLACE VIEW view_batch_jobs_latest AS 
SELECT batch_job, MAX(lastrun_timestamp_end) AS lastrun_timestamp_end
FROM batch_jobs 
GROUP BY batch_job;
-- ===== 025_remove_batch_job_unique_constraint.sql =====
-- Remove UNIQUE constraint from batch_job column in batch_jobs table
ALTER TABLE batch_jobs
DROP CONSTRAINT IF EXISTS batch_jobs_batch_job_key; 
-- ===== 026_reset_batch_jobs_sequence.sql =====
-- Reset the batch_jobs sequence to match the maximum batch_job_id
SELECT setval('batch_jobs_batch_job_id_seq', (SELECT MAX(batch_job_id) FROM batch_jobs), true); 
-- ===== 027_create_cities_table.sql =====
-- Create the cities table to store geographical boundaries of cities
CREATE TABLE IF NOT EXISTS cities (
    city_id   SERIAL PRIMARY KEY,                   -- Unique identifier for each city, auto-incrementing
    city      VARCHAR(50) NOT NULL UNIQUE,          -- Name of the city
    geom        GEOMETRY(MultiPolygon, 4326) NOT NULL -- Spatial geometry of the city boundary in WGS84 (EPSG:4326)
);

-- Add spatial index for faster geospatial queries
CREATE INDEX IF NOT EXISTS idx_cities_geom
ON cities
USING GIST (geom);
-- ===== 029_add_city_id_to_events_store.sql =====
-- Add city_id column to events_store table
ALTER TABLE events_store
ADD COLUMN city_id INTEGER,
ADD FOREIGN KEY (city_id)
    REFERENCES cities(city_id);

-- Add index for city_id to optimize city-based queries
CREATE INDEX IF NOT EXISTS idx_events_store_city_id
ON events_store(city_id);

-- ===== 030_truncate_batch_jobs_and_events_store.sql =====
/*
 * Migration: Truncate batch_jobs and events_store tables
 * Description: This migration will truncate both batch_jobs and events_store tables
 * and reset their identity sequences. This is a destructive operation that will
 * permanently delete all data from these tables.
 * 
 * Impact: 
 * - All data in batch_jobs and events_store tables will be permanently deleted
 * - Identity sequences (serial columns) will be reset to 1
 * - Any tables with foreign key references to these tables will also be truncated
 *   due to the CASCADE option
 */

-- Truncate both tables and reset their identity sequences
TRUNCATE TABLE batch_jobs, events_store RESTART IDENTITY CASCADE; 
-- ===== 031_reset_and_update_event_types.sql =====
/*
 * Migration: Reset and update event_types table
 * Description: This migration will truncate the event_types table, reset its identity sequence,
 * and insert new event types. This is a destructive operation that will permanently delete
 * all existing event types.
 * 
 * Impact: 
 * - All existing event types will be permanently deleted
 * - Identity sequence will be reset to 1
 * - New event types will be inserted
 */

-- Truncate the event_types table and reset its identity sequence
TRUNCATE TABLE event_types RESTART IDENTITY CASCADE;

-- Insert the new event types
INSERT INTO event_types (event_type)
VALUES
    ('Wildfire'),
    ('Hail'),
    ('Rainfall'),
    ('Earthquake'),
    ('Hurricane'),
    ('High Wind'); 
-- ===== 032_drop_unique_constraint_events_store.sql =====
/*
 * Migration: Drop unique constraint from events_store table
 * Description: This migration removes the unique constraint that prevents duplicate events
 * of the same type at the same location on the same date.
 * 
 * Impact: 
 * - Removes the constraint 'unique_event_type_id_geom_date' from events_store table
 * - After this change, multiple events of the same type can exist at the same location
 *   on the same date
 */

-- Drop the unique constraint if it exists
ALTER TABLE events_store 
DROP CONSTRAINT IF EXISTS unique_event_type_id_geom_date; 
-- ===== 033_update_view_events_store.sql =====
DROP VIEW IF EXISTS view_events_store;

CREATE VIEW view_events_store AS
WITH point_data AS (
    SELECT 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        es.is_active,
        ST_PointOnSurface(
            ST_Intersection(
                ST_Collect(es.geom),
                c.geom
            )
        ) as representative_point,
        COUNT(*) as event_count
    FROM events_store es
    INNER JOIN event_types et 
        ON es.event_type_id = et.event_type_id
    INNER JOIN counties c 
        ON es.county_id = c.county_id
    INNER JOIN batch_jobs bj 
        ON es.batch_job_id = bj.batch_job_id
    WHERE es.is_active = true
    GROUP BY 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        es.is_active,
        c.geom
)
SELECT 
    gen_random_uuid() as event_store_id,
    pd.event_type,
    pd.event_date,
    pd.is_active,
    CASE 
        WHEN pd.event_type IN ('High Wind', 'Rainfall', 'Hail') THEN
            json_build_object(
                'min_value', MIN(NULLIF((es.description::json->>'value'), '')::numeric),
                'max_value', MAX(NULLIF((es.description::json->>'value'), '')::numeric),
                'units', MODE() WITHIN GROUP (ORDER BY NULLIF((es.description::json->>'units'), ''))
            )::text
        ELSE
            (SELECT es2.description 
             FROM events_store es2 
             WHERE ST_Equals(es2.geom, pd.representative_point)
             AND es2.is_active = true)
    END as description,
    MAX(pd.event_count) as event_count,
    pd.representative_point as geom,
    pd.county
FROM point_data pd
INNER JOIN events_store es 
    ON es.is_active = true
    AND es.county_id = pd.county_id
    AND es.event_type_id = pd.event_type_id
GROUP BY 
    pd.event_type,
    pd.event_date,
    pd.is_active,
    pd.representative_point,
    pd.county;
-- ===== 035_make_county_id_nullable.sql =====
/*
 * Migration: Make county_id nullable in events_store table
 * Description: This migration makes the county_id column nullable to handle hurricane events
 * that occur outside of Texas counties. Events outside Texas will have NULL county_id.
 */

-- Make county_id nullable
ALTER TABLE events_store 
ALTER COLUMN county_id DROP NOT NULL;

-- ===== 036_create_view_hurricanes.sql =====
/*
 * Migration: Create separate hurricane view
 * Description: This migration creates a dedicated view for hurricane events to handle
 * the special case of hurricanes outside Texas boundaries and simplify the main view.
 */

-- Create dedicated hurricane view
DROP VIEW IF EXISTS view_hurricanes;
CREATE VIEW view_hurricanes AS
SELECT 
    es.event_store_id,
    et.event_type,
    es.event_date,
    es.is_active,
    es.description,
    es.geom,
    c.county
FROM events_store es
INNER JOIN event_types et 
    ON es.event_type_id = et.event_type_id
LEFT JOIN counties c 
    ON es.county_id = c.county_id
WHERE es.is_active = true
AND et.event_type = 'Hurricane'; 
-- ===== 037_update_view_events_store.sql =====
/*
 * Migration: Simplify main view by excluding hurricanes
 * Description: This migration updates the main view_events_store to exclude hurricanes
 * since they now have their own dedicated view.
 */

DROP VIEW IF EXISTS view_events_store;

CREATE VIEW view_events_store AS
WITH point_data AS (
    SELECT 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        es.is_active,
        CASE
            WHEN ST_IsEmpty(ST_Intersection(ST_Collect(es.geom), c.geom))
                THEN ST_PointOnSurface(ST_Collect(es.geom))
            ELSE
                ST_PointOnSurface(ST_Intersection(ST_Collect(es.geom), c.geom))
        END as representative_point,
        COUNT(*) as event_count
    FROM events_store es
    INNER JOIN event_types et 
        ON es.event_type_id = et.event_type_id
    INNER JOIN counties c 
        ON es.county_id = c.county_id
    INNER JOIN batch_jobs bj 
        ON es.batch_job_id = bj.batch_job_id
    WHERE es.is_active = true
    AND et.event_type != 'Hurricane'  -- Exclude hurricanes
    GROUP BY 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        es.is_active,
        c.geom
)
SELECT 
    gen_random_uuid() as event_store_id,
    pd.event_type,
    pd.event_date,
    pd.is_active,
    CASE 
        WHEN pd.event_type IN ('High Wind', 'Rainfall', 'Hail') THEN
            json_build_object(
                'min_value', MIN(NULLIF((es.description::json->>'value'), '')::numeric),
                'max_value', MAX(NULLIF((es.description::json->>'value'), '')::numeric),
                'units', MODE() WITHIN GROUP (ORDER BY NULLIF((es.description::json->>'units'), ''))
            )::text
        ELSE
            (SELECT es2.description 
             FROM events_store es2 
             WHERE ST_Equals(es2.geom, pd.representative_point)
             AND es2.is_active = true)
    END as description,
    MAX(pd.event_count) as event_count,
    pd.representative_point as geom,
    pd.county
FROM point_data pd
INNER JOIN events_store es 
    ON es.is_active = true
    AND es.county_id = pd.county_id
    AND es.event_type_id = pd.event_type_id
GROUP BY 
    pd.event_type,
    pd.event_date,
    pd.is_active,
    pd.representative_point,
    pd.county;
-- ===== 039_update_view_events_store.sql =====
DROP VIEW IF EXISTS view_events_store;

CREATE VIEW view_events_store AS
WITH point_data AS (
    SELECT 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        MIN(es.created_at) as created_at,
        es.is_active,
        CASE
            WHEN ST_IsEmpty(ST_Intersection(ST_Collect(es.geom), c.geom))
                THEN ST_PointOnSurface(ST_Collect(es.geom))
            ELSE
                ST_PointOnSurface(ST_Intersection(ST_Collect(es.geom), c.geom))
        END as representative_point,
        COUNT(*) as event_count
    FROM events_store es
    INNER JOIN event_types et 
        ON es.event_type_id = et.event_type_id
    INNER JOIN counties c 
        ON es.county_id = c.county_id
    INNER JOIN batch_jobs bj 
        ON es.batch_job_id = bj.batch_job_id
    WHERE es.is_active = true
    AND et.event_type != 'Hurricane'  -- Exclude hurricanes
    GROUP BY 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        es.is_active,
        c.geom
)
SELECT 
    gen_random_uuid() as event_store_id,
    pd.event_type,
    pd.event_date,
    MIN(pd.created_at) as created_at,
    pd.is_active,
    CASE 
        WHEN pd.event_type IN ('High Wind', 'Rainfall', 'Hail') THEN
            json_build_object(
                'min_value', MIN(NULLIF((es.description::json->>'value'), '')::numeric),
                'max_value', MAX(NULLIF((es.description::json->>'value'), '')::numeric),
                'units', MODE() WITHIN GROUP (ORDER BY NULLIF((es.description::json->>'units'), ''))
            )::text
        ELSE
            (SELECT es2.description 
             FROM events_store es2 
             WHERE ST_Equals(es2.geom, pd.representative_point)
             AND es2.is_active = true)
    END as description,
    MAX(pd.event_count) as event_count,
    pd.representative_point as geom,
    pd.county
FROM point_data pd
INNER JOIN events_store es 
    ON es.is_active = true
    AND es.county_id = pd.county_id
    AND es.event_type_id = pd.event_type_id
GROUP BY 
    pd.event_type,
    pd.event_date,
    pd.is_active,
    pd.representative_point,
    pd.county;
-- ===== 040_update_view_hurricanes.sql =====
DROP VIEW IF EXISTS view_hurricanes;

CREATE VIEW view_hurricanes AS
SELECT 
    es.event_store_id,
    et.event_type,
    es.event_date,
    es.created_at,
    es.is_active,
    es.description,
    es.geom,
    c.county
FROM events_store es
INNER JOIN event_types et 
    ON es.event_type_id = et.event_type_id
LEFT JOIN counties c 
    ON es.county_id = c.county_id
WHERE es.is_active = true
AND et.event_type = 'Hurricane'; 
-- ===== 041_modify_event_date =====
DROP VIEW IF EXISTS view_hurricanes;
DROP VIEW IF EXISTS view_events_store;

ALTER TABLE events_store ALTER COLUMN event_date TYPE timestamp;

CREATE VIEW view_events_store AS
WITH point_data AS (
    SELECT 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        MIN(es.created_at) as created_at,
        es.is_active,
        CASE
            WHEN ST_IsEmpty(ST_Intersection(ST_Collect(es.geom), c.geom))
                THEN ST_PointOnSurface(ST_Collect(es.geom))
            ELSE
                ST_PointOnSurface(ST_Intersection(ST_Collect(es.geom), c.geom))
        END as representative_point,
        COUNT(*) as event_count
    FROM events_store es
    INNER JOIN event_types et 
        ON es.event_type_id = et.event_type_id
    INNER JOIN counties c 
        ON es.county_id = c.county_id
    INNER JOIN batch_jobs bj 
        ON es.batch_job_id = bj.batch_job_id
    WHERE es.is_active = true
    AND et.event_type != 'Hurricane'  -- Exclude hurricanes
    GROUP BY 
        c.county,
        c.county_id,
        et.event_type,
        et.event_type_id,
        es.event_date,
        es.is_active,
        c.geom
)
SELECT 
    gen_random_uuid() as event_store_id,
    pd.event_type,
    pd.event_date,
    MIN(pd.created_at) as created_at,
    pd.is_active,
    CASE 
        WHEN pd.event_type IN ('High Wind', 'Rainfall', 'Hail') THEN
            json_build_object(
                'min_value', MIN(NULLIF((es.description::json->>'value'), '')::numeric),
                'max_value', MAX(NULLIF((es.description::json->>'value'), '')::numeric),
                'units', MODE() WITHIN GROUP (ORDER BY NULLIF((es.description::json->>'units'), ''))
            )::text
        ELSE
            (SELECT es2.description 
             FROM events_store es2 
             WHERE ST_Equals(es2.geom, pd.representative_point)
             AND es2.is_active = true)
    END as description,
    MAX(pd.event_count) as event_count,
    pd.representative_point as geom,
    pd.county
FROM point_data pd
INNER JOIN events_store es 
    ON es.is_active = true
    AND es.county_id = pd.county_id
    AND es.event_type_id = pd.event_type_id
GROUP BY 
    pd.event_type,
    pd.event_date,
    pd.is_active,
    pd.representative_point,
    pd.county;

CREATE VIEW view_hurricanes AS
SELECT 
    es.event_store_id,
    et.event_type,
    es.event_date,
    es.created_at,
    es.is_active,
    es.description,
    es.geom,
    c.county
FROM events_store es
INNER JOIN event_types et 
    ON es.event_type_id = et.event_type_id
LEFT JOIN counties c 
    ON es.county_id = c.county_id
WHERE es.is_active = true
AND et.event_type = 'Hurricane'; 

grant select on view_hurricanes to ed_app_admin;
grant select on view_events_store to ed_app_admin;
grant select on view_hurricanes to ed_app_user;
grant select on view_events_store to ed_app_user;
