# Recall — Role Attribution + Session Naming + Deletion Fix

Based on full audit of the `semantic-memory` library source code.

---

## Key library findings that affect these fixes

### 1. `Role` enum already serializes lowercase

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System, User, Assistant, Tool,
}
```

The BUG-03 fix (normalizing role casing in `sessions.svelte.ts`) was **unnecessary** — the library already serializes as `"user"` and `"assistant"`. Harmless but redundant.

### 2. `SearchSource::Message` carries role info

```rust
pub enum SearchSource {
    Message {
        message_id: i64,
        session_id: String,
        role: String,        // ← "user", "assistant", etc.
    },
    // ...
}
```

The role is available in every message search result. Recall's `chat.rs` **ignores it**.

### 3. No `rename_session` exists in the library

`MemoryStore` exposes `create_session(&str)` and `delete_session(&str)`. There is no method to update a session's channel/metadata after creation.

### 4. `delete_session` doesn't clean up HNSW entries

`MemoryStore::delete_session()` calls `conversation::delete_session()` which does:
- Clean up FTS entries for all messages in the session
- `DELETE FROM sessions WHERE id = ?1` (CASCADE deletes messages + rowid maps)

But it does NOT remove `msg:N` keys from the HNSW index. After deletion, the HNSW graph still contains vector entries for those messages. On future searches, HNSW may return these keys, but the SQL resolution (`SELECT ... WHERE id IN (...)`) finds nothing and silently skips them. Over time, orphaned entries accumulate, degrading search quality by occupying result slots with phantom hits.

---

## FIX A — Role Attribution (LLM confuses who said what)

### Root cause

In `chat.rs`, all search results — facts, chunks, and messages — are built into the system prompt with identical formatting:

```
[2] (message  score: 0.850)
The Eiffel Tower was built in 1889.
```

The LLM has no way to distinguish whether that message was said by the user or by the assistant. Since the system prompt says "you have access to the user's stored knowledge," the LLM attributes everything to the user.

### Fix — `src-tauri/src/commands/chat.rs`

**Replace the `full_sources` construction:**

Find:
```rust
    // Full content for LLM context
    let full_sources: Vec<SearchResultSer> = results
        .iter()
        .map(|r| {
            let source_type = match &r.source {
                semantic_memory::SearchSource::Fact { .. } => "fact",
                semantic_memory::SearchSource::Chunk { .. } => "document",
                semantic_memory::SearchSource::Message { .. } => "message",
            };
            SearchResultSer {
                content: r.content.clone(),
                source_type: source_type.to_string(),
                score: r.score,
            }
        })
        .collect();
```

Replace with:
```rust
    // Full content for LLM context — messages get role-prefixed
    let full_sources: Vec<SearchResultSer> = results
        .iter()
        .map(|r| {
            let (source_type, content) = match &r.source {
                semantic_memory::SearchSource::Fact { .. } => {
                    ("fact", r.content.clone())
                }
                semantic_memory::SearchSource::Chunk { .. } => {
                    ("document", r.content.clone())
                }
                semantic_memory::SearchSource::Message { role, .. } => {
                    let role_label = match role.as_str() {
                        "user" => "User",
                        "assistant" => "Assistant",
                        _ => "System",
                    };
                    ("message", format!("[{}]: {}", role_label, r.content))
                }
            };
            SearchResultSer {
                content,
                source_type: source_type.to_string(),
                score: r.score,
            }
        })
        .collect();
```

**Update the system prompt instructions:**

Find in `build_system_prompt`:
```rust
    let mut prompt = String::from(
        "You are Recall, a private personal memory assistant. You have access to the user's \
         stored knowledge — facts they've saved, documents they've ingested, and past conversations.\n\n\
         When relevant context is provided below, use it naturally in your response. Never say \
         \"according to my memory\" or \"I found in my database\" — just use the information as if \
         you naturally know it, the way a knowledgeable friend would.\n\n\
         If no relevant context is found, respond helpfully based on general knowledge. \
         Be concise, warm, and direct.\n",
    );
```

Replace with:
```rust
    let mut prompt = String::from(
        "You are Recall, a private personal memory assistant. You have access to the user's \
         stored knowledge — facts they've saved, documents they've ingested, and past conversations.\n\n\
         When relevant context is provided below, use it naturally in your response. Never say \
         \"according to my memory\" or \"I found in my database\" — just use the information as if \
         you naturally know it, the way a knowledgeable friend would.\n\n\
         Message results are labeled [User] or [Assistant] to indicate who said them. \
         Pay attention to these labels — do not attribute your own previous statements to the user, \
         and do not attribute the user's statements to yourself.\n\n\
         If no relevant context is found, respond helpfully based on general knowledge. \
         Be concise, warm, and direct.\n",
    );
```

### Verification

Retrieved messages now appear in the LLM context as:
```
[2] (message  score: 0.850)
[User]: Tell me about the Eiffel Tower

[3] (message  score: 0.820)
[Assistant]: The Eiffel Tower was built in 1889 and stands 324 meters tall.
```

The LLM can now correctly distinguish who said what.

---

## FIX B — Session Naming

### Root cause

`create_session` in the Tauri command always passes `"recall"`:
```rust
let id = state.memory.create_session("recall").await.map_err(|e| e.to_string())?;
```

The library has no `rename_session` method, so there's no way to update the name after the first message.

### Fix — Two parts

#### Part 1: Library addition — `semantic-memory`

**File:** `src/conversation.rs` — add at the end:

```rust
/// Update a session's channel (display name).
pub fn rename_session(
    conn: &Connection,
    session_id: &str,
    new_channel: &str,
) -> Result<(), MemoryError> {
    let affected = conn.execute(
        "UPDATE sessions SET channel = ?1, updated_at = datetime('now') WHERE id = ?2",
        params![new_channel, session_id],
    )?;
    if affected == 0 {
        return Err(MemoryError::SessionNotFound(session_id.to_string()));
    }
    Ok(())
}
```

**File:** `src/lib.rs` — add to `impl MemoryStore`:

```rust
    /// Rename a session's channel (display name).
    pub async fn rename_session(&self, session_id: &str, new_channel: &str) -> Result<(), MemoryError> {
        let sid = session_id.to_string();
        let ch = new_channel.to_string();
        self.with_conn(move |conn| conversation::rename_session(conn, &sid, &ch))
            .await
    }
```

#### Part 2: Recall app — auto-name sessions after first message

**File:** `src-tauri/src/commands/session.rs` — add new command:

```rust
#[tauri::command]
pub async fn rename_session(
    session_id: String,
    name: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state
        .memory
        .rename_session(&session_id, &name)
        .await
        .map_err(|e| e.to_string())
}
```

**File:** `src-tauri/src/commands/mod.rs` — already exports session module, no change needed.

**File:** `src-tauri/src/lib.rs` — add to invoke_handler:

```rust
    commands::session::rename_session,
```

**File:** `src/lib/commands.ts` — add:

```ts
  renameSession: (sessionId: string, name: string) =>
    invoke<void>('rename_session', { session_id: sessionId, name }),
```

**File:** `src/lib/stores/chat.svelte.ts` — after the first message in a new session, rename it.

Find the block inside `sendMessage` where we refresh sidebar metadata:

```ts
    // Refresh sidebar metadata (session message count, memory stats)
    try {
      const sessionsStore = getSessionsStore();
      const memoryStore = getMemoryStore();
      sessionsStore.loadSessions();
      memoryStore.refreshStats();
    } catch (_) {
      // Non-critical — sidebar will catch up on next interaction
    }
```

Replace with:

```ts
    // Refresh sidebar metadata (session message count, memory stats)
    try {
      const sessionsStore = getSessionsStore();
      const memoryStore = getMemoryStore();

      // Auto-name session after first user message
      if (messages.length <= 2) {
        // This is the first exchange (user + assistant = 2 messages)
        const sessionName = content.trim().slice(0, 50) + (content.trim().length > 50 ? '…' : '');
        try {
          await commands.renameSession(sessionsStore.activeSessionId, sessionName);
        } catch (_) {
          // Non-critical
        }
      }

      sessionsStore.loadSessions();
      memoryStore.refreshStats();
    } catch (_) {
      // Non-critical — sidebar will catch up on next interaction
    }
```

### Verification

After the first message in a new session, the sidebar shows the first 50 characters of the user's message instead of "Chat · Today 2:35 PM."

---

## FIX C — Session Deletion

### Root cause analysis

The deletion chain is now:
1. `commands.deleteSession(sessionId)` → `{ session_id: sessionId }` (fixed in V1) ✓
2. Rust `delete_session` → `state.memory.delete_session(&session_id)` ✓
3. Library → FTS cleanup + CASCADE delete in transaction ✓
4. Frontend → `sessions.filter(...)` removes from local state ✓

The SQL deletion is correct. **Two issues remain:**

**Issue 1: Errors are silently swallowed** — if the backend throws for any reason, the user sees nothing. The session stays in the list.

**Issue 2: HNSW orphans accumulate** — deleted messages leave phantom entries in the HNSW graph. Not a crash, but degrades search quality over time.

### Fix — Part 1: Surface deletion errors (Recall app)

**File:** `src/lib/stores/sessions.svelte.ts`

Find:
```ts
async function deleteSession(sessionId: string) {
  try {
    await commands.deleteSession(sessionId);
    sessions = sessions.filter((s) => s.id !== sessionId);
    if (activeSessionId === sessionId && sessions.length > 0) {
      await switchSession(sessions[0].id);
    }
  } catch (err) {
    console.error('Failed to delete session:', err);
  }
}
```

Replace with:
```ts
async function deleteSession(sessionId: string) {
  await commands.deleteSession(sessionId);
  sessions = sessions.filter((s) => s.id !== sessionId);
  if (activeSessionId === sessionId && sessions.length > 0) {
    await switchSession(sessions[0].id);
  }
}
```

And update the Sidebar handler to catch and display:

**File:** `src/lib/components/Sidebar.svelte`

Find:
```ts
  async function handleDeleteSession(e: Event, sessionId: string) {
    e.stopPropagation();
    await sessions.deleteSession(sessionId);
  }
```

Replace with:
```ts
  async function handleDeleteSession(e: Event, sessionId: string) {
    e.stopPropagation();
    try {
      await sessions.deleteSession(sessionId);
    } catch (err) {
      console.error('Failed to delete session:', err);
      // Reload to ensure consistency
      await sessions.loadSessions();
    }
  }
```

### Fix — Part 2: HNSW cleanup on session delete (Library change)

**File:** `semantic-memory/src/lib.rs`

Find:
```rust
    /// Delete a session and all its messages.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        let session_id = session_id.to_string();
        self.with_conn(move |conn| conversation::delete_session(conn, &session_id))
            .await
    }
```

Replace with:
```rust
    /// Delete a session and all its messages.
    ///
    /// Cleans up HNSW entries for embedded messages before deleting from SQLite.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        // Collect message IDs for HNSW cleanup before CASCADE delete removes them
        #[cfg(feature = "hnsw")]
        let message_ids: Vec<i64> = {
            let sid = session_id.to_string();
            self.with_conn(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id FROM messages WHERE session_id = ?1 AND embedding IS NOT NULL"
                )?;
                let ids = stmt
                    .query_map(rusqlite::params![sid], |row| row.get(0))?
                    .collect::<Result<Vec<i64>, _>>()?;
                Ok(ids)
            })
            .await?
        };

        // Delete session (CASCADE handles messages + FTS cleanup inside transaction)
        let sid = session_id.to_string();
        self.with_conn(move |conn| conversation::delete_session(conn, &sid))
            .await?;

        // Clean up HNSW entries for deleted messages
        #[cfg(feature = "hnsw")]
        {
            let guard = self.inner.hnsw_index.read().unwrap();
            for msg_id in &message_ids {
                let key = format!("msg:{}", msg_id);
                if let Err(e) = guard.delete(&key) {
                    tracing::warn!("Failed to remove HNSW entry for {}: {}", key, e);
                }
            }
        }

        Ok(())
    }
```

### Verification

1. Deleting a session now removes all associated HNSW entries, preventing phantom search results.
2. If the backend delete fails, the error propagates to the Sidebar handler, which reloads the session list to stay consistent.

---

## Summary of all changes

### Recall app (`src-tauri/` and `src/`)

| File | Change |
|------|--------|
| `src-tauri/src/commands/chat.rs` | Role-prefix messages in full_sources + update system prompt |
| `src-tauri/src/commands/session.rs` | Add `rename_session` command |
| `src-tauri/src/lib.rs` | Register `rename_session` in invoke_handler |
| `src/lib/commands.ts` | Add `renameSession` |
| `src/lib/stores/chat.svelte.ts` | Auto-name session after first message |
| `src/lib/stores/sessions.svelte.ts` | Remove try/catch from `deleteSession` (let errors propagate) |
| `src/lib/components/Sidebar.svelte` | Catch delete errors + reload session list |

### semantic-memory library

| File | Change |
|------|--------|
| `src/conversation.rs` | Add `rename_session()` function |
| `src/lib.rs` | Add `MemoryStore::rename_session()` public method |
| `src/lib.rs` | Add HNSW cleanup to `MemoryStore::delete_session()` |
