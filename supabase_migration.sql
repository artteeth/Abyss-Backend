-- 在 Supabase SQL Editor 中执行此脚本以创建 memories 表

create table if not exists memories (
  id          bigserial primary key,
  user_id     text        not null,
  role        text        not null,          -- 'user' | 'assistant' | 'system'
  content     text        not null,
  summary     boolean     default null,      -- true 表示这条是压缩摘要
  created_at  timestamptz default now()
);

-- 按用户 + 时间查询加速
create index if not exists idx_memories_user_created
  on memories (user_id, created_at);

-- 按用户 + summary 查询加速
create index if not exists idx_memories_user_summary
  on memories (user_id, summary);
