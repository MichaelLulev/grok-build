use super::*;
use crate::scrollback::block::RenderBlock;
use crate::scrollback::blocks::BtwBlock;

fn btw_notif(session: &str, question: &str, answer: &str, asked_at: &str) -> AcpClientMessage {
    let payload = serde_json::json!({
        "sessionId": session,
        "update": {
            "sessionUpdate": "btw",
            "question": question,
            "answer": answer,
            "asked_at": asked_at,
        },
        "_meta": { "isReplay": true }
    });
    let raw = serde_json::value::to_raw_value(&payload).unwrap();
    let (tx, _rx) = tokio::sync::oneshot::channel();
    AcpClientMessage::ExtNotification(xai_acp_lib::AcpArgs {
        request: acp::ExtNotification::new("x.ai/session/update", raw.into()),
        response_tx: tx,
    })
}

#[test]
fn replay_btw_pushes_collapsed_block() {
    let mut app = make_app_with_agent("sess-A");
    app.agents
        .get_mut(&AgentId(0))
        .unwrap()
        .session
        .loading_replay = true;
    let affected = handle(
        btw_notif(
            "sess-A",
            "what is rust",
            "a **systems** language",
            "2026-08-20T12:00:00Z",
        ),
        &mut app,
    );
    assert!(affected, "Btw for the active agent must request a redraw");

    let agent = app.agents.get(&AgentId(0)).unwrap();
    assert_eq!(agent.scrollback.len(), 1);
    match &agent.scrollback.get(0).unwrap().block {
        RenderBlock::Btw(BtwBlock { question, .. }) => {
            assert_eq!(question, "what is rust");
        }
        other => panic!("expected Btw, got {other:?}"),
    }
    let content = match &agent.scrollback.get(0).unwrap().block {
        RenderBlock::Btw(block) => block.content().text(),
        _ => unreachable!(),
    };
    assert!(
        content.contains("systems"),
        "answer must round-trip into the block"
    );
}

#[test]
fn replay_btw_stamps_created_at_from_asked_at() {
    let mut app = make_app_with_agent("sess-A");
    app.agents
        .get_mut(&AgentId(0))
        .unwrap()
        .session
        .loading_replay = true;
    assert!(handle(
        btw_notif(
            "sess-A",
            "earlier",
            "a1",
            "2026-08-20T12:00:00Z",
        ),
        &mut app,
    ));
    let agent = app.agents.get(&AgentId(0)).unwrap();
    let created = agent
        .scrollback
        .get(0)
        .unwrap()
        .created_at
        .expect("asked_at");
    let expected = chrono::DateTime::parse_from_rfc3339("2026-08-20T12:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    assert_eq!(created.with_timezone(&chrono::Utc), expected);
}

#[test]
fn replay_btw_does_not_count_as_full_reconnect_replay() {
    let mut app = make_app_with_agent("sess-A");
    let id = AgentId(0);
    {
        let agent = app.agents.get_mut(&id).unwrap();
        agent
            .scrollback
            .push_block(RenderBlock::user_prompt("keep me"));
        agent
            .scrollback
            .push_block(RenderBlock::Btw(BtwBlock::new("aside", "answer")));
        agent.begin_session_reload(1);
    }
    assert!(handle(
        btw_notif(
            "sess-A",
            "aside",
            "answer",
            "2026-08-20T12:00:00Z",
        ),
        &mut app,
    ));
    let agent = app.agents.get_mut(&id).unwrap();
    assert!(agent.finish_session_reload(1, true));
    let texts: Vec<_> = agent
        .scrollback
        .iter_entries()
        .filter_map(|(_, entry)| match &entry.block {
            RenderBlock::UserPrompt(p) => Some(p.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        texts.iter().any(|t| *t == "keep me"),
        "reconnect stash must survive a replay-only /btw inject, got {texts:?}"
    );
    let btw: Vec<_> = agent
        .scrollback
        .iter_entries()
        .filter_map(|(_, entry)| match &entry.block {
            RenderBlock::Btw(block) => Some((block.question.as_str(), block.content().text())),
            _ => None,
        })
        .collect();
    assert_eq!(
        btw,
        vec![("aside", "answer".to_string())],
        "keep-stash merge must not duplicate the live /btw pin"
    );
}
