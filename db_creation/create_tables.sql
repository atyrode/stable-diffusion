CREATE TABLE IF NOT EXISTS sd_results
(
    result_id       serial PRIMARY KEY,
    prompt          TEXT    NOT NULL,
    image_file_name VARCHAR NOT NULL,
    generation_date DATE    NOT NULL,
    sent_to_discord BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS sd_score
(
    result_id SERIAL,
    score     NUMERIC,
    CONSTRAINT fk_sd_results
        FOREIGN KEY (result_id)
            REFERENCES sd_results (result_id)

);
