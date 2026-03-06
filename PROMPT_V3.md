# Task: Fix Role Attribution, Session Naming, and Deletion in Recall

Read `BUGFIXES_V3.md` in this directory before writing any code. It contains every fix with exact before/after code.

Also read `CLAUDE.md` and `BUILD_SPEC.md` for architecture context.

## Ground rules

1. Only change what's specified in BUGFIXES_V3.md.
2. Changes span BOTH the Recall app AND the semantic-memory library. Library changes are explicitly marked as "LIBRARY CHANGE."
3. Preserve all existing CSS.
4. Every fix has a verification note.

## Fix order

1. FIX-A → `src-tauri/src/commands/chat.rs` (role attribution in RAG context)
2. FIX-B1 → `semantic-memory/src/conversation.rs` + `semantic-memory/src/lib.rs` (LIBRARY: add rename_session)
3. FIX-B2 → `src-tauri/src/commands/session.rs` + `src-tauri/src/lib.rs` + `src/lib/commands.ts` + `src/lib/stores/chat.svelte.ts` (auto-name sessions)
4. FIX-C1 → `src/lib/stores/sessions.svelte.ts` + `src/lib/components/Sidebar.svelte` (surface delete errors)
5. FIX-C2 → `semantic-memory/src/lib.rs` (LIBRARY: HNSW cleanup on session delete)

After all fixes, run `cargo check` from both the Recall app root and the semantic-memory library root.
