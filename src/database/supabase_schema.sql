-- Run this exactly as is in the Supabase SQL Editor (https://supabase.com/dashboard)
CREATE TABLE aic_captions (
    id SERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    keyframe_name TEXT NOT NULL,
    shot_id INTEGER,
    caption TEXT,
    shot_summary TEXT,
    video_summary TEXT,
    transcript TEXT,
    objects TEXT,
    -- This is the Postgres Full-Text Search tsvector column
    fts tsvector GENERATED ALWAYS AS (
        to_tsvector('english', 
            coalesce(caption, '') || ' ' || 
            coalesce(shot_summary, '') || ' ' || 
            coalesce(video_summary, '') || ' ' || 
            coalesce(transcript, '') || ' ' || 
            coalesce(objects, '')
        )
    ) STORED
);

-- Create a fast GIN index for lightning-fast text search
CREATE INDEX aic_captions_fts_idx ON aic_captions USING GIN (fts);

-- Create standard indexes for filtering
CREATE INDEX aic_captions_video_id_idx ON aic_captions (video_id);
