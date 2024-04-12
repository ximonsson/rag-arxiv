CREATE OR REPLACE MACRO text_clean(txt) AS trim(regexp_replace(txt, '\s+', ' ', 'g'));

CREATE OR REPLACE VIEW arxiv_filtered AS
	SELECT * FROM arxiv WHERE
		regexp_matches(categories, '(cs.LG|cs.CL|cs.AI|cs.CV|cs.DS)') AND
		v1_created_date >= '2010-01-01';

CREATE OR REPLACE TABLE document AS
	SELECT
		*,
		lower(
			printf('%s, written by %s. %s', title, authors, abstract)
		) AS body
	FROM (
		SELECT
			id,
			hash(id) AS eid,
			text_clean(title) AS title,
			text_clean(abstract) AS abstract,
			authors_parsed_cat AS authors,
			submitter,
			v1_created_date,
			vlast_created_date,
			update_date,
			categories,
		FROM
			arxiv_filtered
	);
