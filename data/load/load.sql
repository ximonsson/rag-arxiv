CREATE OR REPLACE TABLE arxiv AS
	SELECT
		id,
		submitter,
		authors,
		title,
		categories,
		abstract,
		len(versions) AS n_ver,
		list_aggr (
			list_transform(
				authors_parsed,
				x -> trim(list_aggr(list_reverse(x), 'string_agg', ' '))
			),
			'string_agg',
			', '
		) AS authors_parsed_cat,
		strptime(
			versions[1]['created'], '%a, %d %b %Y %H:%M:%S %Z'
		) AS v1_created_date,
		strptime(
			versions[-1]['created'], '%a, %d %b %Y %H:%M:%S %Z'
		) AS vlast_created_date,
		update_date,
	FROM (
		SELECT * FROM read_json(
			getenv('ARXIV_RAW_JSON_FP'), auto_detect=true, format='auto'
		)
	);

COPY arxiv TO '/home/ximon/data/arxiv.parquet' (FORMAT 'parquet');
