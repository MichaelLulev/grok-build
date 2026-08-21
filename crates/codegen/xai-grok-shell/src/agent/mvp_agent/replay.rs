//! Forwards recorded session updates back to a loading client, fitting
//! completion records written before the size limit existed.

use std::collections::VecDeque;
use std::path::PathBuf;

use agent_client_protocol as acp;
use xai_grok_paths::AbsPathBuf;

use super::{MvpAgent, mark_as_replay, stamp_meta_value};
use crate::session::persistence::BtwEntry;
use crate::session::storage::{
    ReplayToolCollapser, jsonl_envelope_timestamp_secs, rewind_discarded_time_windows,
};

/// Successful `/btw` entries from `btw_history.jsonl` next to `updates.jsonl`.
fn load_successful_btw_entries(updates_path: &std::path::Path) -> Vec<BtwEntry> {
    let Some(dir) = updates_path.parent() else {
        return Vec::new();
    };
    load_successful_btw_from_str(
        &std::fs::read_to_string(dir.join("btw_history.jsonl")).unwrap_or_default(),
    )
}

fn load_successful_btw_from_str(contents: &str) -> Vec<BtwEntry> {
    let mut entries: Vec<BtwEntry> = contents
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            serde_json::from_str::<BtwEntry>(line).ok()
        })
        .filter(|e| e.success && !e.question.trim().is_empty() && !e.answer.trim().is_empty())
        .collect();
    entries.sort_by_key(|e| e.asked_at);
    entries
}

fn filter_btw_surviving_rewinds(mut entries: Vec<BtwEntry>, raw: &str) -> Vec<BtwEntry> {
    let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
    let windows = rewind_discarded_time_windows(&lines);
    if windows.is_empty() {
        return entries;
    }
    entries.retain(|e| {
        let t = e.asked_at.timestamp();
        !windows.iter().any(|&(start, end)| t >= start && t <= end)
    });
    entries
}

fn load_btw_entries_for_replay(updates_path: &std::path::Path, raw: &str) -> Vec<BtwEntry> {
    filter_btw_surviving_rewinds(load_successful_btw_entries(updates_path), raw)
}

/// Whether a `/btw` asked at `asked_at` should be injected before this
/// `updates.jsonl` line (or at end-of-replay flush). Using ask time — not
/// dismiss time — keeps resume order matching the live call site.
fn btw_due_for_line(
    asked_at: chrono::DateTime<chrono::Utc>,
    line_ts: Option<i64>,
    flush_rest: bool,
) -> bool {
    flush_rest || line_ts.is_some_and(|ts| asked_at.timestamp() <= ts)
}

/// Max in-flight `forward_with_completion` receivers during cold resume.
/// Unbounded enqueue + sync pager apply peaks the pager at multi-GB on huge
/// sessions; this keeps ACP apply roughly windowed.
pub(super) const REPLAY_COMPLETION_WINDOW: usize = 64;

type ReplayCompletionRx = tokio::sync::oneshot::Receiver<xai_acp_lib::AcpResult<()>>;

/// Sliding window of replay completion receivers. Awaits the oldest when full
/// so at most [`REPLAY_COMPLETION_WINDOW`] notifications sit un-acked.
pub(super) struct ReplayCompletionDrain {
    pending: VecDeque<ReplayCompletionRx>,
    forwarded: usize,
}

impl ReplayCompletionDrain {
    pub(super) fn new() -> Self {
        Self {
            pending: VecDeque::with_capacity(REPLAY_COMPLETION_WINDOW),
            forwarded: 0,
        }
    }

    pub(super) async fn push(&mut self, rx: ReplayCompletionRx) {
        if self.pending.len() >= REPLAY_COMPLETION_WINDOW
            && let Some(oldest) = self.pending.pop_front()
        {
            let _ = oldest.await;
        }
        self.pending.push_back(rx);
        self.forwarded += 1;
    }

    pub(super) fn forwarded(&self) -> usize {
        self.forwarded
    }

    pub(super) async fn drain_all(mut self) {
        while let Some(rx) = self.pending.pop_front() {
            let _ = rx.await;
        }
    }
}

impl MvpAgent {
    /// Records written before completions were bounded can still be too long
    /// for a client to read. `None` drops one that cannot be shrunk, which
    /// costs a completion event but keeps the connection.
    fn fitted_replay_params(
        params: Box<serde_json::value::RawValue>,
    ) -> Option<Box<serde_json::value::RawValue>> {
        use crate::tools::task_completed_frame::{Refit, refit_recorded};

        match refit_recorded(&params) {
            Refit::Unchanged => Some(params),
            Refit::Fitted(fitted) => Some(fitted.into_inner()),
            Refit::Unfittable => {
                tracing::warn!(
                    bytes = params.get().len(),
                    "replay: dropping a completion too long to send"
                );
                None
            }
        }
    }

    /// Forward one raw JSONL replay line. Returns the completion receiver when
    /// a notification was actually sent.
    ///
    /// Dispatches by on-disk method name:
    /// - ACP updates (`"session/update"`) → typed `SessionNotification` for correct
    ///   TUI dispatch (direct dispatch preserves Rust types, not method strings).
    /// - xAI updates (`"_x.ai/session/update"`) → `ExtNotification`.
    ///
    /// When `mark_replay` is true, the notification is tagged with
    /// `_meta.isReplay: true` so the client knows it's historical data.
    /// Cursor-based reconnects set this to false for events after the cursor
    /// so the client processes them as live updates.
    pub(super) fn forward_raw_replay_line(
        &self,
        line: &str,
        persist_data: Option<&serde_json::Value>,
        target_client_id: Option<&serde_json::Value>,
        mark_replay: bool,
        collapser: &mut ReplayToolCollapser,
    ) -> Option<ReplayCompletionRx> {
        use crate::session::storage::RawLinePeek;

        let env = match serde_json::from_str::<RawLinePeek<'_>>(line) {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!(?e, "replay: skipping unparseable JSONL line");
                return None;
            }
        };
        // updates.jsonl only persists `_x.ai/session/update` and `session/update`.
        // Unknown methods fall through to the ACP parse below and are dropped on error.
        let method = env.method.unwrap_or("session/update");
        let Some(raw_params) = env.params else {
            tracing::debug!("replay: skipping JSONL line with no params");
            return None;
        };
        let is_xai = method == "_x.ai/session/update";

        if is_xai {
            // The fast-path forwards raw params with no `_meta` round-trip, so it
            // can stamp nothing. When a `target_client_id` is present we MUST take
            // the injection path instead, otherwise the replay would lose the
            // target and the leader would broadcast it to every subscriber.
            if target_client_id.is_none() && !mark_replay {
                if let Ok(owned) =
                    serde_json::value::RawValue::from_string(raw_params.get().to_owned())
                    && let Some(owned) = Self::fitted_replay_params(owned)
                {
                    return Some(
                        self.gateway
                            .forward_with_completion(acp::ExtNotification::new(
                                "x.ai/session/update",
                                std::sync::Arc::from(owned),
                            )),
                    );
                }
                return None;
            }
            let Ok(mut params) = serde_json::from_str::<serde_json::Value>(raw_params.get()) else {
                tracing::debug!("replay: skipping xAI update with unparseable params");
                return None;
            };
            if let Some(obj) = params.as_object_mut() {
                let meta = obj.entry("_meta").or_insert_with(|| serde_json::json!({}));
                if let Some(m) = meta.as_object_mut() {
                    // `isReplay` only applies to historical replay events, not the
                    // post-cursor live deltas that reach this path when a target is set.
                    if mark_replay {
                        m.insert("isReplay".to_string(), serde_json::json!(true));
                    }
                    if let Some(pd) = persist_data {
                        m.insert("x.ai/persist".to_string(), pd.clone());
                    }
                    if let Some(tid) = target_client_id {
                        m.insert("x.ai/leaderClientId".to_string(), tid.clone());
                    }
                }
            }
            if let Ok(raw_val) = serde_json::value::to_raw_value(&params)
                && let Some(raw_val) = Self::fitted_replay_params(raw_val)
            {
                return Some(
                    self.gateway
                        .forward_with_completion(acp::ExtNotification::new(
                            "x.ai/session/update",
                            std::sync::Arc::from(raw_val),
                        )),
                );
            }
            return None;
        }

        let Ok(notification) = serde_json::from_str::<acp::SessionNotification>(raw_params.get())
        else {
            tracing::debug!("replay: skipping ACP update with unparseable params");
            return None;
        };
        let acp::SessionNotification {
            session_id,
            update,
            meta,
            ..
        } = notification;
        let update = collapser.push(update)?;
        let mut notification = acp::SessionNotification::new(session_id, update);
        notification.meta = meta;
        if mark_replay {
            mark_as_replay(&mut notification.meta, persist_data);
        }
        // Stamp the leader unicast target regardless of mark_replay so the
        // leader routes both historical and post-cursor live deltas only to
        // the loading client.
        if let Some(tid) = target_client_id {
            stamp_meta_value(&mut notification.meta, "x.ai/leaderClientId", tid);
        }
        Some(self.gateway.forward_with_completion(notification))
    }

    fn forward_replay_btw(
        &self,
        session_id: &acp::SessionId,
        entry: &BtwEntry,
        persist_data: Option<&serde_json::Value>,
        target_client_id: Option<&serde_json::Value>,
    ) -> Option<ReplayCompletionRx> {
        let mut meta = serde_json::Map::new();
        meta.insert("isReplay".into(), serde_json::json!(true));
        if let Some(pd) = persist_data {
            meta.insert("x.ai/persist".into(), pd.clone());
        }
        if let Some(tid) = target_client_id {
            meta.insert("x.ai/leaderClientId".into(), tid.clone());
        }
        let notification = crate::extensions::notification::SessionNotification {
            session_id: session_id.clone(),
            update: crate::extensions::notification::SessionUpdate::Btw {
                question: entry.question.clone(),
                answer: entry.answer.clone(),
                asked_at: entry.asked_at.to_rfc3339(),
            },
            meta: Some(serde_json::Value::Object(meta)),
        };
        let params = serde_json::value::to_raw_value(&notification).ok()?;
        Some(
            self.gateway
                .forward_with_completion(acp::ExtNotification::new(
                    "x.ai/session/update",
                    std::sync::Arc::from(params),
                )),
        )
    }

    async fn emit_btw_due(
        &self,
        session_id: &acp::SessionId,
        pending: &mut std::iter::Peekable<std::vec::IntoIter<BtwEntry>>,
        line_ts: Option<i64>,
        persist_data: Option<&serde_json::Value>,
        target_client_id: Option<&serde_json::Value>,
        drain: &mut ReplayCompletionDrain,
        flush_rest: bool,
    ) {
        loop {
            let Some(next) = pending.peek() else {
                return;
            };
            let due = btw_due_for_line(next.asked_at, line_ts, flush_rest);
            if !due {
                return;
            }
            let entry = pending.next().expect("peeked");
            if let Some(rx) =
                self.forward_replay_btw(session_id, &entry, persist_data, target_client_id)
            {
                drain.push(rx).await;
            }
        }
    }

    /// Replay updates from disk and drain completions.
    /// Returns `(initial_total_tokens, end_offset, unfinished_subagents)`.
    pub(super) async fn replay_session_updates(
        &self,
        session_id: &acp::SessionId,
        cwd: &AbsPathBuf,
        updates_file_path: &Option<PathBuf>,
        persist_data: Option<&serde_json::Value>,
        target_client_id: Option<&serde_json::Value>,
        cursor: Option<&str>,
    ) -> Result<(u64, u64, Vec<(String, String)>), acp::Error> {
        let mut replay_timer = crate::instrumentation_timer!("session.load_session_replay");
        replay_timer.with_field("session_id", session_id.0.as_ref());
        replay_timer.with_field("cwd", cwd.as_str());

        let Some(updates_path) = updates_file_path.as_ref() else {
            tracing::warn!(session_id = %session_id.0, "replay: no updates file path");
            return Ok((0, 0, Vec::new()));
        };

        let file_size = std::fs::metadata(updates_path)
            .map(|m| m.len())
            .unwrap_or(0);

        // Inline blocking I/O: spawn_blocking has multi-second latency on LocalSet.
        let raw_contents = match std::fs::read_to_string(updates_path) {
            Ok(s) if !s.is_empty() => s,
            _ => {
                if cursor.is_none() {
                    let mut drain = ReplayCompletionDrain::new();
                    let mut pending = load_successful_btw_entries(updates_path)
                        .into_iter()
                        .peekable();
                    self.emit_btw_due(
                        session_id,
                        &mut pending,
                        None,
                        persist_data,
                        target_client_id,
                        &mut drain,
                        true,
                    )
                    .await;
                    drain.drain_all().await;
                }
                return Ok((0, 0, Vec::new()));
            }
        };
        let end_offset = raw_contents.len() as u64;

        let mut prepared = {
            let _timer = crate::instrumentation_timer!("session.replay.read_and_filter");
            crate::session::storage::prepare_replay_lines(&raw_contents, cursor)
        };
        let unfinished_subagents = std::mem::take(&mut prepared.unfinished_subagents);

        if cursor.is_some() {
            let sending = prepared.lines.len();
            if prepared.mark_replay {
                tracing::warn!(
                    session_id = %session_id.0,
                    "replay: cursor not found, falling back to full replay"
                );
            } else {
                tracing::info!(
                    session_id = %session_id.0,
                    skipped = prepared.total_live - sending,
                    remaining = sending,
                    "replay: cursor found, skipping events"
                );
            }
        }

        let last_tokens = prepared.last_tokens;
        let mark_replay = prepared.mark_replay;

        if let Some(max_seq) = prepared.max_event_seq {
            crate::util::event_id::ensure_event_counter_at_least(max_seq + 1);
        }

        let lines_to_send = prepared.lines;
        let updates_count = lines_to_send.len() as u64;
        let mut drain = ReplayCompletionDrain::new();
        let mut btw_pending = if mark_replay {
            load_btw_entries_for_replay(updates_path, &raw_contents)
                .into_iter()
                .peekable()
        } else {
            Vec::new().into_iter().peekable()
        };

        {
            let _timer = crate::instrumentation_timer!("session.replay.forward_updates");
            let mut collapser = ReplayToolCollapser::new();
            for line in &lines_to_send {
                self.emit_btw_due(
                    session_id,
                    &mut btw_pending,
                    jsonl_envelope_timestamp_secs(line),
                    persist_data,
                    target_client_id,
                    &mut drain,
                    false,
                )
                .await;
                if let Some(rx) = self.forward_raw_replay_line(
                    line,
                    persist_data,
                    target_client_id,
                    mark_replay,
                    &mut collapser,
                ) {
                    drain.push(rx).await;
                }
            }
            self.emit_btw_due(
                session_id,
                &mut btw_pending,
                None,
                persist_data,
                target_client_id,
                &mut drain,
                true,
            )
            .await;
            // Do not flush collapser leftovers: synthesizing a ToolCall here
            // would drop the persisted `_meta.eventId` and duplicate on
            // incremental reconnect. Child stream EOF flush is separate.
        }

        if updates_count > 0 && drain.forwarded() == 0 {
            tracing::warn!(
                updates_count,
                "Replay sent updates but collected 0 completions — \
                 forward_raw_replay_line must use gateway.forward_with_completion(). \
                 See: session/load notification ordering bug."
            );
        }
        {
            let _timer = crate::instrumentation_timer!("session.replay.drain_completions");
            drain.drain_all().await;
        }

        tracing::info!(
            session_id = %session_id.0,
            updates_count,
            end_offset,
            file_size,
            "replay: completed"
        );

        replay_timer.with_field("updates_count", updates_count);

        Ok((last_tokens, end_offset, unfinished_subagents))
    }

    /// Enqueue replay notifications for updates appended after `from_offset`.
    /// Returns completion receivers; callers open the gate then drain.
    /// Intentionally sync (not async) so no prompt-task progress before gate flip.
    ///
    /// The delta tail is typically small (appends during the just-finished
    /// replay). Windowing would require `.await` here and would delay the gate
    /// flip; the caller drains the returned receivers before `LoadSessionResponse`.
    ///
    /// When `mark_replay` is false (cursor-based reconnect), delta events are
    /// forwarded without `_meta.isReplay` since they are truly new events the
    /// client has not seen.
    pub(super) fn replay_session_updates_from_offset_enqueue(
        &self,
        session_id: &acp::SessionId,
        updates_file_path: &Option<PathBuf>,
        from_offset: u64,
        persist_data: Option<&serde_json::Value>,
        target_client_id: Option<&serde_json::Value>,
        mark_replay: bool,
    ) -> Vec<ReplayCompletionRx> {
        use std::io::{Read, Seek, SeekFrom};

        let Some(updates_path) = updates_file_path.as_ref() else {
            return Vec::new();
        };

        let mut file = match std::fs::File::open(updates_path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        if file.seek(SeekFrom::Start(from_offset)).is_err() {
            return Vec::new();
        }
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_err() || contents.is_empty() {
            return Vec::new();
        }

        let live_lines = crate::session::storage::filter_delta_replay_lines(&contents);
        let delta_count = live_lines.len();

        let mut completions = Vec::with_capacity(live_lines.len());
        let mut collapser = ReplayToolCollapser::new();
        for line in &live_lines {
            if let Some(rx) = self.forward_raw_replay_line(
                line,
                persist_data,
                target_client_id,
                mark_replay,
                &mut collapser,
            ) {
                completions.push(rx);
            }
        }

        if delta_count > 0 && completions.is_empty() {
            tracing::warn!(
                delta_count,
                "Delta replay sent updates but collected 0 completions — \
                 forward_raw_replay_line must use gateway.forward_with_completion(). \
                 See: session/load notification ordering bug."
            );
        }

        if delta_count > 0 {
            tracing::info!(
                session_id = %session_id.0,
                delta_count,
                from_offset,
                "Delta replay enqueued updates (drain pending)"
            );
        }

        completions
    }
}

#[cfg(test)]
mod drain_tests {
    use super::{REPLAY_COMPLETION_WINDOW, ReplayCompletionDrain};

    #[tokio::test]
    async fn completion_window_drains_all_ready_receivers() {
        assert_eq!(REPLAY_COMPLETION_WINDOW, 64);
        let mut drain = ReplayCompletionDrain::new();
        for _ in 0..5 {
            let (tx, rx) = tokio::sync::oneshot::channel();
            tx.send(Ok(())).unwrap();
            drain.push(rx).await;
        }
        assert_eq!(drain.forwarded(), 5);
        drain.drain_all().await;
    }

    #[tokio::test]
    async fn completion_window_awaits_oldest_before_exceeding_cap() {
        let mut drain = ReplayCompletionDrain::new();
        let mut early_txs = Vec::new();
        for _ in 0..REPLAY_COMPLETION_WINDOW {
            let (tx, rx) = tokio::sync::oneshot::channel();
            early_txs.push(tx);
            drain.push(rx).await;
        }
        let (overflow_tx, overflow_rx) =
            tokio::sync::oneshot::channel::<xai_acp_lib::AcpResult<()>>();
        {
            let push = drain.push(overflow_rx);
            tokio::pin!(push);
            tokio::select! {
                _ = &mut push => panic!("push must wait for the oldest completion when the window is full"),
                _ = tokio::task::yield_now() => {}
            }
            early_txs
                .remove(0)
                .send(Ok(()))
                .expect("oldest receiver still live");
            push.await;
        }
        overflow_tx.send(Ok(())).ok();
        for tx in early_txs {
            let _ = tx.send(Ok(()));
        }
        drain.drain_all().await;
    }

    #[tokio::test]
    async fn drain_all_awaits_fifo_remainder() {
        let mut drain = ReplayCompletionDrain::new();
        let (tx0, rx0) = tokio::sync::oneshot::channel();
        let (tx1, rx1) = tokio::sync::oneshot::channel();
        let (tx2, rx2) = tokio::sync::oneshot::channel();
        drain.push(rx0).await;
        drain.push(rx1).await;
        drain.push(rx2).await;
        let drain_all = drain.drain_all();
        tokio::pin!(drain_all);
        tokio::select! {
            _ = &mut drain_all => panic!("drain_all must wait for index 0 first"),
            _ = tokio::task::yield_now() => {}
        }
        tx2.send(Ok(())).ok();
        tokio::select! {
            _ = &mut drain_all => panic!("index 2 ready must not finish drain before 0"),
            _ = tokio::task::yield_now() => {}
        }
        tx0.send(Ok(())).unwrap();
        tx1.send(Ok(())).unwrap();
        drain_all.await;
    }
}

#[cfg(test)]
mod btw_replay_tests {
    use super::{
        btw_due_for_line, filter_btw_surviving_rewinds, load_successful_btw_from_str,
    };
    use crate::session::storage::jsonl_envelope_timestamp_secs;
    use crate::session::persistence::BtwEntry;
    use chrono::{TimeZone, Utc};

    fn line(q: &str, a: &str, success: bool, asked: i64) -> String {
        serde_json::to_string(&BtwEntry {
            btw_session_id: "btw-1".into(),
            parent_session_id: "s".into(),
            asked_at: Utc.timestamp_opt(asked, 0).single().unwrap(),
            question: q.into(),
            answer: a.into(),
            model: "grok".into(),
            success,
            error: None,
            attempts: 1,
        })
        .unwrap()
    }

    #[test]
    fn load_skips_failed_and_empty() {
        let jsonl = [
            line("ok q", "ok a", true, 10),
            line("fail q", "", true, 11),
            line("err q", "nope", false, 12),
            line("", "ans", true, 13),
            line("later", "yes", true, 14),
        ]
        .join("\n");
        let got = load_successful_btw_from_str(&jsonl);
        let qs: Vec<_> = got.iter().map(|e| e.question.as_str()).collect();
        assert_eq!(qs, ["ok q", "later"]);
    }

    #[test]
    fn load_sorts_by_asked_at() {
        let jsonl = [line("b", "2", true, 20), line("a", "1", true, 10)].join("\n");
        let got = load_successful_btw_from_str(&jsonl);
        assert_eq!(got[0].question, "a");
        assert_eq!(got[1].question, "b");
    }

    #[test]
    fn btw_due_at_or_before_line_timestamp_not_dismiss_time() {
        let asked = Utc.timestamp_opt(15, 0).single().unwrap();
        assert!(
            !btw_due_for_line(asked, Some(10), false),
            "a later ask must wait for a later conversation event"
        );
        assert!(
            btw_due_for_line(asked, Some(15), false),
            "inject at the first event at or after the ask"
        );
        assert!(btw_due_for_line(asked, Some(20), false));
        assert!(
            !btw_due_for_line(asked, None, false),
            "unknown line time must not emit early"
        );
        assert!(btw_due_for_line(asked, None, true));
    }

    #[test]
    fn timestamp_peek_seconds_millis_rfc3339() {
        assert_eq!(
            jsonl_envelope_timestamp_secs(
                r#"{"timestamp":1787218575,"method":"session/update","params":{}}"#
            ),
            Some(1787218575)
        );
        assert_eq!(
            jsonl_envelope_timestamp_secs(
                r#"{"timestamp":1787218575000,"method":"session/update","params":{}}"#
            ),
            Some(1787218575)
        );
        let rfc = "2026-08-20T12:00:00Z";
        let expected = chrono::DateTime::parse_from_rfc3339(rfc)
            .unwrap()
            .timestamp();
        assert_eq!(
            jsonl_envelope_timestamp_secs(&format!(
                r#"{{"timestamp":"{rfc}","method":"session/update","params":{{}}}}"#
            )),
            Some(expected)
        );
    }

    #[test]
    fn rewind_dead_branch_btw_is_not_restored() {
        let raw = [
            r#"{"timestamp":10,"method":"session/update","params":{"sessionId":"s","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"first"}}}}"#,
            r#"{"timestamp":11,"method":"session/update","params":{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"r1"}}}}"#,
            r#"{"timestamp":20,"method":"session/update","params":{"sessionId":"s","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"second"}}}}"#,
            r#"{"timestamp":21,"method":"session/update","params":{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"r2"}}}}"#,
            r#"{"timestamp":25,"method":"_x.ai/session/update","params":{"sessionId":"s","update":{"sessionUpdate":"rewind_marker","target_prompt_index":1,"created_at":"2024-01-01"}}}"#,
            r#"{"timestamp":30,"method":"session/update","params":{"sessionId":"s","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"replacement"}}}}"#,
        ]
        .join("\n");
        let jsonl = [
            line("kept", "a", true, 11),
            line("dead", "b", true, 21),
            line("after", "c", true, 30),
        ]
        .join("\n");
        let entries = load_successful_btw_from_str(&jsonl);
        let got = filter_btw_surviving_rewinds(entries, &raw);
        let qs: Vec<_> = got.iter().map(|e| e.question.as_str()).collect();
        assert_eq!(qs, ["kept", "after"]);
    }
}
