-- Preserve the setup event trigger; remove unnecessary API caller privileges.
revoke execute on function public.rls_auto_enable() from public, anon, authenticated;
