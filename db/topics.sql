-- all words in all documents
CREATE OR REPLACE TEMP VIEW all_words AS
WITH cte AS (
	SELECT
		id,
		unnest(regexp_split_to_array(lower(abstract), '[\W\s]+')) AS word,
		topic
	FROM doc
)
SELECT *
FROM cte
ANTI JOIN fts_main_doc.stopwords AS a ON a.sw = cte.word
WHERE word <> '';

-- frequency of all words in all documents
CREATE OR REPLACE TEMP VIEW T AS
SELECT word, count(*) AS v FROM all_words GROUP BY word;

-- frequency of all words within a cluster
CREATE OR REPLACE TEMP VIEW ti AS
SELECT
	topic,
	word,
	count(*) AS v,
FROM all_words
GROUP BY topic, word;

-- number of words per cluster
CREATE OR REPLACE TEMP VIEW W as
SELECT topic, sum(v) AS wi FROM ti GROUP BY topic;

-- tf-idf
-- ti / wi x log (m / sum(t_j))
CREATE OR REPLACE TEMP VIEW tf_idf AS
SELECT
	*,
	ti.v / wi * log((SELECT count(*) FROM doc) / T.v) AS score
FROM ti
JOIN T USING (word)
JOIN W USING (topic);

--QUALIFY row_number() OVER (PARTITION BY topic ORDER BY tf_idf DESC) < 11;
