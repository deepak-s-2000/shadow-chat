import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas.chat import SessionCreate, SessionResponse, MessageResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    session = models.Session(title=body.title or "New Chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("", response_model=list[SessionResponse])
def list_sessions(db: Session = Depends(get_db)):
    return db.query(models.Session).order_by(models.Session.updated_at.desc()).all()


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    from app.core.vectorstore import vector_store
    vector_store.delete_session(session_id)


@router.get("/{session_id}/stats")
def get_stats(session_id: str, db: Session = Depends(get_db)):
    if not db.query(models.Session).filter(models.Session.id == session_id).first():
        raise HTTPException(status_code=404, detail="Session not found")

    rows = (
        db.query(models.Turn)
        .filter(models.Turn.session_id == session_id)
        .order_by(models.Turn.created_at.asc())
        .all()
    )

    turns = []
    for i, r in enumerate(rows):
        try:
            llm_log = json.loads(r.llm_call_log)
        except Exception:
            llm_log = []
        try:
            tool_execs = json.loads(r.tool_executions)
        except Exception:
            tool_execs = []

        tools_called = [ex["name"] for ex in tool_execs]
        llm_calls = len(llm_log)

        turns.append({
            "turn": i + 1,
            "user_message": r.user_message[:200],
            "mode": r.mode,
            "actual_input_tokens": r.total_tokens_in,
            "actual_output_tokens": r.total_tokens_out,
            "actual_total_tokens": r.total_tokens_in + r.total_tokens_out,
            "estimated_saved_tokens": r.estimated_saved_tokens,
            "llm_calls": llm_calls,
            "tools_called": tools_called,
            "llm_call_log": llm_log,
            "created_at": r.created_at.isoformat(),
        })

    total_input = sum(r.total_tokens_in for r in rows)
    total_output = sum(r.total_tokens_out for r in rows)
    total_saved = sum(r.estimated_saved_tokens for r in rows)
    total_llm_calls = sum(len(json.loads(r.llm_call_log or "[]")) for r in rows)

    return {
        "session_id": session_id,
        "summary": {
            "total_actual_input_tokens": total_input,
            "total_actual_output_tokens": total_output,
            "total_actual_tokens": total_input + total_output,
            "total_estimated_saved_tokens": total_saved,
            "total_llm_calls": total_llm_calls,
            "turns": len(turns),
        },
        "turns": turns,
    }


@router.get("/{session_id}/history", response_model=list[MessageResponse])
def get_history(session_id: str, db: Session = Depends(get_db)):
    if not db.query(models.Session).filter(models.Session.id == session_id).first():
        raise HTTPException(status_code=404, detail="Session not found")
    return (
        db.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.asc())
        .all()
    )
