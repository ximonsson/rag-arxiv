using JSON3, Dates, TimeZones, DataFrames, Parquet

df = JSON3.read("/home/ximon/data/arxiv-metadata-oai-snapshot.json", jsonlines = true) |> DataFrame

version_date(v) = Date(v["created"], dateformat"e, d u y H:M:S Z")

transform!(
	df,
	:versions =>
		ByRow(x -> x[1] |> version_date, x[-1] |> version_date) =>
		[:v1_created_date, :vlast_created_date],
	:versions => length => :n_versions,
	:authors_parsed => ByRow(x -> join(join.(x, " ") .|> strip, "|") => :authors_parsed_cat),
)

select!(df, Not([:authors_parsed, :versions]))
