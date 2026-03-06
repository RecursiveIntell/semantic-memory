# BUGFIXES_V3.md — Role Attribution, Session Naming, Deletion

---

## FIX-A: Role attribution in RAG context

### Root cause
Search results from past messages include a `role` field ("user" or "assistant") in `SearchSource::Message`, but `chat.rs` ignores it. All retrieved messages are fed to the LLM as unlabeled text. The LLM attributes everything to the user.

### File: `src-tauri/src/commands/chat.rs`

**Find the full_sources construction:**

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

**Replace with:**

```rust
    // Full content for LLM context — messages get role-prefixed for attribution
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

**Find the system prompt in `build_system_prompt`:**

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

**Replace with:**

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
Retrieved messages now appear as `[User]: ...` or `[Assistant]: ...` in the LLM context. The system prompt explicitly instructs the model to respect these labels.

---

## FIX-B1: Add `rename_session` to semantic-memory (LIBRARY CHANGE)

### Root cause
The library has no way to update a session's channel/name after creation. Every session stays named whatever was passed to `create_session`.

### File: `semantic-memory/src/conversation.rs`

**Add at the end of the file, before the closing (or after the last function):**

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

### File: `semantic-memory/src/lib.rs`

**Find the Session Management section:**

```rust
    // ─── Session Management ─────────────────────────────────────

    /// Create a new conversation session. Returns the session ID (UUID v4).
    pub async fn create_session(&self, channel: &str) -> Result<String, MemoryError> {
        let channel = channel.to_string();
        self.with_conn(move |conn| conversation::create_session(conn, &channel, None))
            .await
    }

    /// List recent sessions, newest first.
    pub async fn list_sessions(
```

**Add `rename_session` after `create_session`:**

```rust
    // ─── Session Management ─────────────────────────────────────

    /// Create a new conversation session. Returns the session ID (UUID v4).
    pub async fn create_session(&self, channel: &str) -> Result<String, MemoryError> {
        let channel = channel.to_string();
        self.with_conn(move |conn| conversation::create_session(conn, &channel, None))
            .await
    }

    /// Rename a session's channel (display name).
    pub async fn rename_session(&self, session_id: &str, new_channel: &str) -> Result<(), MemoryError> {
        let sid = session_id.to_string();
        let ch = new_channel.to_string();
        self.with_conn(move |conn| conversation::rename_session(conn, &sid, &ch))
            .await
    }

    /// List recent sessions, newest first.
    pub async fn list_sessions(
```

### Verification
`MemoryStore::rename_session("session-uuid", "My cool chat")` updates the `channel` field in SQLite and bumps `updated_at`.

---

## FIX-B2: Auto-name sessions in Recall app

### File: `src-tauri/src/commands/session.rs`

**Add this command after `delete_session`:**

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

### File: `src-tauri/src/lib.rs`

**Find in the invoke_handler:**

```rust
            commands::session::get_active_session,
```

**Add after it:**

```rust
            commands::session::rename_session,
```

### File: `src/lib/commands.ts`

**Find:**

```ts
  getActiveSession: () => invoke<string>('get_active_session'),
```

**Add after it:**

```ts
  renameSession: (sessionId: string, name: string) =>
    invoke<void>('rename_session', { session_id: sessionId, name }),
```

### File: `src/lib/stores/chat.svelte.ts`

**Find the sidebar refresh block inside the `try` in `sendMessage`:**

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

**Replace with:**

```ts
    // Refresh sidebar metadata (session message count, memory stats)
    try {
      const sessionsStore = getSessionsStore();
      const memoryStore = getMemoryStore();

      // Auto-name session after first user message
      if (messages.length <= 2) {
        const sessionName = content.trim().slice(0, 50) + (content.trim().length > 50 ? '…' : '');
        try {
          await commands.renameSession(sessionsStore.activeSessionId, sessionName);
        } catch (_) {
          // Non-critical — session keeps default name
        }
      }

      sessionsStore.loadSessions();
      memoryStore.refreshStats();
    } catch (_) {
      // Non-critical — sidebar will catch up on next interaction
    }
```

### Verification
After sending the first message in a new session, the sidebar shows the first 50 characters of the user's message as the session name. Subsequent messages don't overwrite the name (the `messages.length <= 2` guard ensures this — at that point the array has exactly the user message + assistant response).

---

## FIX-C1: Surface session deletion errors

### Root cause
`deleteSession` in the sessions store wraps everything in try/catch and swallows errors. If the backend delete fails, the user sees nothing — the session stays in the list.

### File: `src/lib/stores/sessions.svelte.ts`

**Find:**

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

**Replace with:**

```ts
async function deleteSession(sessionId: string) {
  await commands.deleteSession(sessionId);
  sessions = sessions.filter((s) => s.id !== sessionId);
  if (activeSessionId === sessionId && sessions.length > 0) {
    await switchSession(sessions[0].id);
  }
}
```

### File: `src/lib/components/Sidebar.svelte`

**Find:**

```ts
  async function handleDeleteSession(e: Event, sessionId: string) {
    e.stopPropagation();
    await sessions.deleteSession(sessionId);
  }
```

**Replace with:**

```ts
  async function handleDeleteSession(e: Event, sessionId: string) {
    e.stopPropagation();
    try {
      await sessions.deleteSession(sessionId);
    } catch (err) {
      console.error('Failed to delete session:', err);
      // Reload from backend to ensure UI stays consistent
      await sessions.loadSessions();
    }
  }
```

### Verification
If delete fails, the error appears in the console AND the session list reloads from the backend to stay consistent. If delete succeeds, behavior is unchanged from before.

---

## FIX-C2: HNSW cleanup on session delete (LIBRARY CHANGE)

### Root cause
`MemoryStore::delete_session()` cleans up FTS entries and CASCADE-deletes messages, but leaves orphaned `msg:N` entries in the HNSW index. These phantom entries accumulate over time, taking up search result slots with ghost hits.

### File: `semantic-memory/src/lib.rs`

**Find:**

```rust
    /// Delete a session and all its messages.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        let session_id = session_id.to_string();
        self.with_conn(move |conn| conversation::delete_session(conn, &session_id))
            .await
    }
```

**Replace with:**

```rust
    /// Delete a session and all its messages.
    ///
    /// Cleans up HNSW entries for embedded messages before CASCADE delete.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        // Collect message IDs with embeddings for HNSW cleanup
        #[cfg(feature = "hnsw")]
        let message_ids: Vec<i64> = {
            let sid = session_id.to_string();
            self.with_conn(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id FROM messages WHERE session_id = ?1 AND embedding IS NOT NULL",
                )?;
                let ids = stmt
                    .query_map(rusqlite::params![sid], |row| row.get(0))?
                    .collect::<Result<Vec<i64>, _>>()?;
                Ok(ids)
            })
            .await?
        };

        // Delete session (CASCADE handles messages, FTS cleanup inside transaction)
        let sid = session_id.to_string();
        self.with_conn(move |conn| conversation::delete_session(conn, &sid))
            .await?;

        // Remove orphaned HNSW entries
        #[cfg(feature = "hnsw")]
        {
            let guard = self.inner.hnsw_index.read().unwrap();
            for msg_id in &message_ids {
                let key = format!("msg:{}", msg_id);
                if let Err(e) = guard.delete(&key) {
                    tracing::warn!("Failed to remove HNSW entry {}: {}", key, e);
                }
            }
        }

        Ok(())
    }
```

### Verification
After deleting a session, `msg:N` entries for that session's messages are removed from the HNSW index. Future searches won't waste result slots on phantom hits from deleted sessions.

---

## Post-implementation checklist

- [ ] Retrieved messages in the LLM context are prefixed with `[User]:` or `[Assistant]:`
- [ ] System prompt instructs the model to respect role labels
- [ ] `semantic-memory` has `rename_session` in both `conversation.rs` and `MemoryStore`
- [ ] `rename_session` Tauri command is registered in `lib.rs` invoke_handler
- [ ] `renameSession` exists in `commands.ts` with correct arg names
- [ ] First message in a new session triggers `renameSession` with truncated content
- [ ] `deleteSession` in sessions store does NOT swallow errors
- [ ] `handleDeleteSession` in Sidebar catches errors and reloads
- [ ] `MemoryStore::delete_session` cleans up HNSW entries before CASCADE delete
- [ ] `cargo check` passes for both `semantic-memory` and `recall` crates
