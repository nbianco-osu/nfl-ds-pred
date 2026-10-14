-- Read-only verification for the current flat-payload snapshot manifest.
-- Normalize numeric scale without rounding; key order matches PostgreSQL JSONB.
WITH normalized AS (
  SELECT r.*, (
    SELECT '{' || string_agg(
      to_jsonb(e.key)::text || ': ' ||
      CASE WHEN jsonb_typeof(e.value) = 'number'
        THEN trim_scale((e.value #>> '{}')::numeric)::text
        ELSE e.value::text
      END, ', ' ORDER BY octet_length(e.key), e.key COLLATE "C"
    ) || '}'
    FROM jsonb_each(r.payload) e
  ) AS canonical
  FROM public.nfl_dataset_records r
)
SELECT dataset, count(*) AS rows,
  encode(sha256(convert_to(string_agg(
    concat_ws('|', id, dataset, snapshot_sha256, record_key,
      coalesce(season::text, ''), coalesce(game_id, ''), md5(canonical)),
    E'\n' ORDER BY id), 'UTF8')), 'hex') AS sha256
FROM normalized
GROUP BY dataset
ORDER BY dataset;

SELECT relrowsecurity AS rls_enabled,
  has_table_privilege('anon', 'public.nfl_dataset_records', 'SELECT') AS anon_read,
  has_table_privilege('authenticated', 'public.nfl_dataset_records', 'SELECT') AS authenticated_read
FROM pg_class
WHERE oid = 'public.nfl_dataset_records'::regclass;
