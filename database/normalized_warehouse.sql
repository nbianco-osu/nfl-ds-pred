-- R22 additive private dimensions and immutable schedule observations.
-- Raw licensed sources remain local; no new buckets or public grants.
create schema nfl_private;
revoke all on schema nfl_private from public, anon, authenticated;

create table nfl_private.teams (
  team_id text primary key,
  name text not null
);
create table nfl_private.venues (
  venue_id text primary key,
  name text not null
);
create table nfl_private.players (
  player_id text primary key,
  name text not null,
  coverage text not null default 'Schedule QB identities only; not complete roster'
);
create table nfl_private.game_observations (
  source_record_id text primary key references public.nfl_dataset_records(id),
  game_id text not null,
  season integer not null,
  home_team text not null references nfl_private.teams(team_id),
  away_team text not null references nfl_private.teams(team_id),
  venue_id text references nfl_private.venues(venue_id),
  home_qb_id text references nfl_private.players(player_id),
  away_qb_id text references nfl_private.players(player_id),
  observed_at timestamptz not null default now(),
  check (home_team <> away_team)
);
create index game_observations_game_idx on nfl_private.game_observations(game_id,season);
create index game_observations_home_idx on nfl_private.game_observations(home_team);
create index game_observations_away_idx on nfl_private.game_observations(away_team);
create index game_observations_venue_idx on nfl_private.game_observations(venue_id);
create index game_observations_home_qb_idx on nfl_private.game_observations(home_qb_id);
create index game_observations_away_qb_idx on nfl_private.game_observations(away_qb_id);
create index nfl_records_dataset_key_idx on public.nfl_dataset_records(dataset,record_key);
alter table nfl_private.teams enable row level security;
alter table nfl_private.venues enable row level security;
alter table nfl_private.players enable row level security;
alter table nfl_private.game_observations enable row level security;
revoke all on all tables in schema nfl_private from public,anon,authenticated;

create function nfl_private.normalize_record(record_id text) returns void
language plpgsql security invoker set search_path = '' as $$
declare r public.nfl_dataset_records; p jsonb; side text; identity text;
begin
  select * into strict r from public.nfl_dataset_records where id=record_id;
  p := r.payload;
  if r.dataset='teams' and nullif(p->>'team_abbr','') is not null then
    insert into nfl_private.teams values(p->>'team_abbr',coalesce(p->>'team_name',p->>'team_abbr'))
      on conflict(team_id) do update set name=excluded.name;
  elsif r.dataset like 'schedule_%' then
    foreach side in array array['home','away'] loop
      identity := nullif(p->>(side || '_team'),'');
      if identity is null then raise exception 'Missing schedule team'; end if;
      insert into nfl_private.teams values(identity,identity) on conflict do nothing;
      identity := nullif(p->>(side || '_qb_id'),'');
      if identity is not null then
        insert into nfl_private.players(player_id,name) values(identity,coalesce(p->>(side || '_qb_name'),identity)) on conflict do nothing;
      end if;
    end loop;
    identity := nullif(p->>'stadium_id','');
    if identity is not null then
      insert into nfl_private.venues values(identity,coalesce(p->>'stadium',identity)) on conflict do nothing;
    end if;
    insert into nfl_private.game_observations(source_record_id,game_id,season,home_team,away_team,venue_id,home_qb_id,away_qb_id)
      values(r.id,r.game_id,r.season,p->>'home_team',p->>'away_team',identity,
        nullif(p->>'home_qb_id',''),nullif(p->>'away_qb_id','')) on conflict do nothing;
  end if;
end;
$$;
revoke execute on function nfl_private.normalize_record(text) from public,anon,authenticated;
create function nfl_private.on_snapshot_insert() returns trigger
language plpgsql security invoker set search_path = '' as $$
begin
  perform nfl_private.normalize_record(new.id);
  return new;
end;
$$;
revoke execute on function nfl_private.on_snapshot_insert() from public,anon,authenticated;
create trigger nfl_normalize_snapshot after insert on public.nfl_dataset_records
  for each row execute function nfl_private.on_snapshot_insert();
