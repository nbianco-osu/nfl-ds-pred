-- Private, immutable snapshot warehouse. Run as project owner.
begin;
create table if not exists public.nfl_dataset_records (
  id text primary key,
  dataset text not null,
  snapshot_sha256 text not null check (length(snapshot_sha256)=64),
  record_key text not null,
  season integer,
  game_id text,
  payload jsonb not null check (jsonb_typeof(payload)='object'),
  imported_at timestamptz not null default now(),
  unique(dataset,snapshot_sha256,record_key)
);
alter table public.nfl_dataset_records enable row level security;
revoke all on public.nfl_dataset_records from anon, authenticated;
grant select, insert on public.nfl_dataset_records to service_role;
create index if not exists nfl_records_game_idx on public.nfl_dataset_records(game_id) where game_id is not null;
create index if not exists nfl_records_dataset_season_idx on public.nfl_dataset_records(dataset,season);
comment on table public.nfl_dataset_records is 'Private source/version-keyed NFL research snapshots. No browser access policies. FTN-derived charted fields: FTN Data via nflverse, CC-BY-SA 4.0.';
commit;

-- Verification: both privileges must be false; RLS must be true.
select relrowsecurity as rls_enabled,
  has_table_privilege('anon','public.nfl_dataset_records','SELECT') as anon_read,
  has_table_privilege('authenticated','public.nfl_dataset_records','SELECT') as authenticated_read
from pg_class where oid='public.nfl_dataset_records'::regclass;
