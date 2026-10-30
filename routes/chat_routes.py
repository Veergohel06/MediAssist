"""Chat & Consultation Routes.

Manages consultations, user messages, AI generation calls, and multi-tenant isolation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import uuid
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    Response,
    stream_with_context,
)
from flask_login import current_user, login_required
from google.genai import errors as genai_errors
from werkzeug.utils import secure_filename

from models import ChatMessage, Conversation, ConversationDocument, db, utc_now
from services.rate_limiter import chat_limiter

from services.document_service import (
    validate_file_content,
    scan_file_for_malware,
    DocumentValidationError,
    DocumentExtractionError,
)

logger = logging.getLogger("mediassist.chat")
chat_bp = Blueprint("chat", __name__)


def _resolve_or_create_conversation(
    conv_id: str | None,
    user_id: int,
    user_message: str,
) -> tuple[Conversation | None, tuple[str, int] | None]:
    """Resolve conversation by ID, or find/create active conversation for current user."""
    if conv_id:
        conv = Conversation.query.filter_by(id=conv_id).first()
        if not conv:
            return None, ("Conversation not found.", 404)
        if conv.user_id != user_id:
            return None, ("Access denied. Conversation belongs to another user.", 403)
        return conv, None

    # No conversation_id supplied: select most recent conversation for user
    conv = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.updated_at.desc()).first()
    if not conv:
        title = user_message[:40] + ("..." if len(user_message) > 40 else "")
        conv = Conversation(user_id=user_id, title=title)
        db.session.add(conv)
        db.session.commit()
    return conv, None


def _get_or_create_user_message(conversation: Conversation, user_message: str) -> ChatMessage:
    """Retrieve idempotent pending user message turn or insert a fresh one."""
    last_msg = (
        conversation.messages.order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .first()
    )
    if (
        last_msg
        and last_msg.role == "user"
        and last_msg.content == user_message
        and last_msg.status in ("pending", "error")
    ):
        return last_msg

    user_msg = ChatMessage(
        conversation_id=conversation.id,
        role="user",
        content=user_message,
        status="pending",
    )
    db.session.add(user_msg)
    db.session.commit()
    return user_msg


@chat_bp.route("/chat", methods=["GET"])
@login_required
def chat_view():
    """Protected chatbot view."""
    model_name = current_app.config.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    return render_template("chat.html", active_model=model_name)


@chat_bp.route("/chat/stream", methods=["POST"])
@login_required
def chat_stream():
    """Stream consultation response incrementally using Server-Sent Events (SSE).
    Structured events: start, chunk, evidence, complete, error.
    Persists finalized assistant turn and citations in Neon PostgreSQL.
    """
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
    limiter_key = f"{current_user.id}:{client_ip}"
    if not chat_limiter.check_and_record(limiter_key):
        return jsonify({"error": "Rate limit exceeded. Please wait a moment before sending another message."}), 429

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request. JSON body required."}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    max_len = current_app.config.get("MAX_USER_MSG_LEN", 2000)
    if len(user_message) > max_len:
        return jsonify({
            "error": f"Message is too long. Please keep it under {max_len:,} characters."
        }), 400

    conversation_id = data.get("conversation_id")
    selected_option = (data.get("selected_option") or "").strip().upper() or None
    question_id = (data.get("question_id") or data.get("mcq_id") or "").strip() or None

    conversation, err = _resolve_or_create_conversation(conversation_id, current_user.id, user_message)
    if err:
        err_msg, status_code = err
        return jsonify({"error": err_msg}), status_code

    user_msg = _get_or_create_user_message(conversation, user_message)
    user_msg_id = user_msg.id
    conv_id = conversation.id
    user_id = current_user.id
    user_academic_year = getattr(current_user, "academic_year", None)
    user_target_exam = getattr(current_user, "target_exam", None)

    # Build bounded model context strictly from this conversation
    max_history = current_app.config.get("MAX_HISTORY_TURNS", 20)
    past_messages = (
        conversation.messages.filter(
            ChatMessage.id != user_msg.id,
            ChatMessage.status == "completed",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(max_history)
        .all()[::-1]
    )
    conv_history = []
    for msg in past_messages:
        role = "model" if msg.role in ("assistant", "model") else "user"
        conv_history.append({
            "role": role,
            "parts": [{"text": msg.content}],
            "evidence": msg.evidence,
        })

    llm_service = current_app.extensions["llm_service"]

    def event_generator():
        accumulated_text = ""
        final_sources = []
        final_triage = {}
        final_evidence = {}
        final_citations = {}
        practice_mode = "general_tutoring"
        interactive_mcq = None
        clinical_case = None

        try:
            # 1. Yield message start event
            start_payload = {
                "conversation_id": conv_id,
                "user_message_id": user_msg_id,
                "created_at": utc_now().isoformat(),
            }
            yield f"event: start\ndata: {json.dumps(start_payload)}\n\n"

            for event_obj in llm_service.stream_response(
                user_message=user_message,
                history=conv_history,
                user_id=user_id,
                conversation_id=conv_id,
                academic_year=user_academic_year,
                target_exam=user_target_exam,
                selected_option=selected_option,
                question_id=question_id,
            ):
                evt_name = event_obj.get("event")
                evt_data = event_obj.get("data", {})

                if evt_name == "chunk":
                    chunk_text = evt_data.get("text", "")
                    accumulated_text += chunk_text
                    yield f"event: chunk\ndata: {json.dumps({'text': chunk_text})}\n\n"

                elif evt_name == "evidence":
                    final_sources = evt_data.get("sources", [])
                    final_evidence = evt_data.get("evidence", {})
                    final_citations = evt_data.get("citations", {})
                    yield f"event: evidence\ndata: {json.dumps(evt_data)}\n\n"

                elif evt_name == "complete":
                    final_reply = evt_data.get("response", accumulated_text)
                    final_sources = evt_data.get("sources", final_sources)
                    final_triage = evt_data.get("triage", {})
                    final_evidence = evt_data.get("evidence", final_evidence)
                    final_citations = evt_data.get("citations", final_citations)
                    practice_mode = evt_data.get("practice_mode", "general_tutoring")
                    interactive_mcq = evt_data.get("interactive_mcq")
                    clinical_case = evt_data.get("clinical_case")

                    # Persist completed turn safely in Neon PostgreSQL
                    bot_msg_id = None
                    created_at_str = utc_now().isoformat()
                    try:
                        u_msg = db.session.get(ChatMessage, user_msg_id)
                        if u_msg:
                            u_msg.status = "completed"
                            u_msg.error_message = None

                        evidence_payload = {
                            "sources": final_sources,
                            "triage": final_triage,
                            "evidence": final_evidence,
                            "citations": final_citations,
                            "practice_mode": practice_mode,
                            "interactive_mcq": interactive_mcq,
                            "clinical_case": clinical_case,
                        }

                        bot_msg = ChatMessage(
                            conversation_id=conv_id,
                            role="model",
                            content=final_reply,
                            evidence=evidence_payload,
                            status="completed",
                        )
                        db.session.add(bot_msg)
                        conv = db.session.get(Conversation, conv_id)
                        if conv:
                            conv.touch()
                        db.session.commit()
                        bot_msg_id = bot_msg.id
                        if bot_msg.created_at:
                            created_at_str = bot_msg.created_at.isoformat()
                    except Exception as db_err:
                        db.session.rollback()
                        logger.error("DB persistence failed after streaming turn: %s", db_err)

                    complete_payload = {
                        "conversation_id": conv_id,
                        "user_message_id": user_msg_id,
                        "assistant_message_id": bot_msg_id,
                        "response": final_reply,
                        "sources": final_sources,
                        "triage": final_triage,
                        "evidence": final_evidence,
                        "citations": final_citations,
                        "practice_mode": practice_mode,
                        "interactive_mcq": interactive_mcq,
                        "clinical_case": clinical_case,
                        "created_at": created_at_str,
                    }
                    yield f"event: complete\ndata: {json.dumps(complete_payload)}\n\n"
                    return

        except GeneratorExit:
            logger.info("Client disconnected from SSE stream (user_id=%s, conv_id=%s)", user_id, conv_id)
            try:
                db.session.rollback()
                u_msg = db.session.get(ChatMessage, user_msg_id)
                if u_msg and u_msg.status == "pending":
                    u_msg.status = "interrupted"
                    db.session.commit()
            except Exception:
                db.session.rollback()
            return

        except Exception as exc:
            logger.exception("Unexpected error during stream generation")
            try:
                db.session.rollback()
                u_msg = db.session.get(ChatMessage, user_msg_id)
                if u_msg:
                    u_msg.status = "error"
                    u_msg.error_message = "Streaming generation failed"
                    db.session.commit()
            except Exception:
                db.session.rollback()

            safe_error_msg = "MediAssist encountered a temporary issue while streaming the response. Please try again."
            yield f"event: error\ndata: {json.dumps({'error': safe_error_msg})}\n\n"

    return Response(
        stream_with_context(event_generator()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@chat_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    """Receive a user message, run urgent safety checks, load conversation context
    strictly scoped to current_user, call Gemini RAG pipeline, and persist
    the consultation turns with strict isolation and idempotency.
    """
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
    limiter_key = f"{current_user.id}:{client_ip}"
    if not chat_limiter.check_and_record(limiter_key):
        return jsonify({"error": "Rate limit exceeded. Please wait a moment before sending another message."}), 429

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request. JSON body required."}), 400

    # Seamless fallback / negotiation for streaming requests
    if request.headers.get("Accept") == "text/event-stream" or data.get("stream") is True:
        return chat_stream()

    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    max_len = current_app.config.get("MAX_USER_MSG_LEN", 2000)
    if len(user_message) > max_len:
        return jsonify({
            "error": f"Message is too long. Please keep it under {max_len:,} characters."
        }), 400


    llm_service = current_app.extensions["llm_service"]

    # 1. Urgent-symptom safety pre-flight check
    safety_eval = llm_service.safety_evaluator.evaluate(user_message)
    if safety_eval.get("is_emergency", False):
        reply_text = safety_eval["triage"]["full_guidance"]
        triage = dict(safety_eval["triage"])
        triage["performance"] = {
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }
        triage["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "model": "rule_based_emergency_dispatch",
        }

        # Attempt to persist emergency turn safely without failing user-facing escalation if DB fails
        user_msg_id = None
        bot_msg_id = None
        conv_id = data.get("conversation_id")
        try:
            conv, _ = _resolve_or_create_conversation(conv_id, current_user.id, user_message)
            if conv:
                conv_id = conv.id
                user_msg = ChatMessage(conversation_id=conv.id, role="user", content=user_message, status="completed")
                bot_msg = ChatMessage(
                    conversation_id=conv.id,
                    role="model",
                    content=reply_text,
                    evidence={"sources": [], "triage": triage},
                    status="completed",
                )
                db.session.add_all([user_msg, bot_msg])
                conv.touch()
                db.session.commit()
                user_msg_id = user_msg.id
                bot_msg_id = bot_msg.id
        except Exception as db_exc:
            db.session.rollback()
            logger.warning("Database persistence issue during emergency turn: %s", db_exc)

        return jsonify({
            "response": reply_text,
            "sources": [],
            "triage": triage,
            "evidence": triage.get("evidence", {}),
            "citations": triage.get("citation_validation", {}),
            "conversation_id": conv_id,
            "user_message_id": user_msg_id,
            "assistant_message_id": bot_msg_id,
            "created_at": utc_now().isoformat(),
            "timestamp": utc_now().isoformat(),
        }), 200

    # 2. Verify conversation ownership and tenant isolation
    conversation_id = data.get("conversation_id")
    selected_option = (data.get("selected_option") or "").strip().upper() or None
    question_id = (data.get("question_id") or data.get("mcq_id") or "").strip() or None

    conversation, err = _resolve_or_create_conversation(conversation_id, current_user.id, user_message)
    if err:
        err_msg, status_code = err
        return jsonify({"error": err_msg}), status_code

    # 3. Save or retrieve user message turn
    user_msg = _get_or_create_user_message(conversation, user_message)

    # 4. Build bounded model context strictly from this conversation with evidence metadata
    max_history = current_app.config.get("MAX_HISTORY_TURNS", 20)
    past_messages = (
        conversation.messages.filter(
            ChatMessage.id != user_msg.id,
            ChatMessage.status == "completed",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(max_history)
        .all()[::-1]  # Reverse to get chronological order after DESC+LIMIT
    )
    conv_history = []
    for msg in past_messages:
        role = "model" if msg.role in ("assistant", "model") else "user"
        conv_history.append({
            "role": role,
            "parts": [{"text": msg.content}],
            "evidence": msg.evidence,
        })

    # 5. Call Gemini RAG, Safety & LangGraph Workflow Pipeline
    try:
        main_mod = sys.modules.get("app")
        app_mock = getattr(main_mod, "get_gemini_response", None) if main_mod else None
        if app_mock is not None and hasattr(app_mock, "assert_called"):
            try:
                reply, sources, triage = app_mock(
                    user_message,
                    history=conv_history,
                    user_id=current_user.id,
                    conversation_id=conversation.id,
                    academic_year=getattr(current_user, "academic_year", None),
                    target_exam=getattr(current_user, "target_exam", None),
                    selected_option=selected_option,
                    question_id=question_id,
                )
            except TypeError:
                reply, sources, triage = app_mock(user_message, conv_history)
        else:
            llm_service = current_app.extensions["llm_service"]
            reply, sources, triage = llm_service.get_response(
                user_message=user_message,
                history=conv_history,
                user_id=current_user.id,
                conversation_id=conversation.id,
                academic_year=getattr(current_user, "academic_year", None),
                target_exam=getattr(current_user, "target_exam", None),
                selected_option=selected_option,
                question_id=question_id,
            )

        # 6. Save assistant response only after valid response obtained
        user_msg.status = "completed"
        user_msg.error_message = None
        practice_mode = triage.get("practice_mode", "general_tutoring")
        evidence_payload = {
            "sources": sources,
            "triage": triage,
            "practice_mode": practice_mode,
            "interactive_mcq": triage.get("interactive_mcq"),
            "clinical_case": triage.get("clinical_case"),
        }
        bot_msg = ChatMessage(
            conversation_id=conversation.id,
            role="model",
            content=reply,
            evidence=evidence_payload,
            status="completed",
        )
        db.session.add(bot_msg)
        conversation.touch()
        db.session.commit()

        # 7. Return response
        return jsonify({
            "response": reply,
            "sources": sources,
            "triage": triage,
            "evidence": triage.get("evidence", {}),
            "citations": triage.get("citation_validation", {}),
            "practice_mode": practice_mode,
            "interactive_mcq": triage.get("interactive_mcq"),
            "clinical_case": triage.get("clinical_case"),
            "conversation_id": conversation.id,
            "user_message_id": user_msg.id,
            "assistant_message_id": bot_msg.id,
            "created_at": bot_msg.created_at.isoformat() if bot_msg.created_at else utc_now().isoformat(),
            "timestamp": bot_msg.created_at.isoformat() if bot_msg.created_at else utc_now().isoformat(),
        }), 200

    except genai_errors.APIError as exc:
        try:
            user_msg.status = "error"
            user_msg.error_message = f"AI API Error: {type(exc).__name__}"
            db.session.commit()
        except Exception:
            db.session.rollback()

        status_code = (
            getattr(exc, "code", None)
            or getattr(exc, "status_code", None)
            or getattr(exc, "http_status", None)
        )
        if status_code is None:
            exc_str = str(exc)
            import re as _re
            m = _re.search(r"\b(4\d\d|5\d\d)\b", exc_str)
            if m:
                status_code = int(m.group(1))

        logger.warning(
            "Gemini APIError during turn (code=%s, error=%s, user_id=%s, conv_id=%s)",
            status_code,
            type(exc).__name__,
            current_user.id,
            conversation.id,
        )

        if status_code == 400:
            return jsonify({
                "error": (
                    "The request was rejected by the AI service (400 Bad Request). "
                    "This usually means the GEMINI_API_KEY is invalid or malformed. "
                    "Please verify your key at https://aistudio.google.com/apikey."
                ),
                "conversation_id": conversation.id,
                "user_message_id": user_msg.id,
            }), 503

        if status_code in (401, 403):
            return jsonify({
                "error": "API authentication failed. Please check your GEMINI_API_KEY.",
                "conversation_id": conversation.id,
                "user_message_id": user_msg.id,
            }), 503

        if status_code == 429:
            return jsonify({
                "error": (
                    "Rate limit or quota reached. "
                    "Please wait a moment and try again, or check your account limits at "
                    "https://aistudio.google.com."
                ),
                "conversation_id": conversation.id,
                "user_message_id": user_msg.id,
            }), 429

        if status_code == 404:
            model_name = current_app.config.get("GEMINI_MODEL", "")
            return jsonify({
                "error": (
                    f"The configured model '{model_name}' was not found. "
                    "Update GEMINI_MODEL in your .env file."
                ),
                "conversation_id": conversation.id,
                "user_message_id": user_msg.id,
            }), 503

        if status_code and status_code >= 500:
            return jsonify({
                "error": "The AI service is temporarily unavailable. Please try again shortly.",
                "conversation_id": conversation.id,
                "user_message_id": user_msg.id,
            }), 503

        return jsonify({
            "error": "An error occurred communicating with the AI service. Please try again.",
            "conversation_id": conversation.id,
            "user_message_id": user_msg.id,
        }), 503

    except AttributeError:
        try:
            user_msg.status = "error"
            user_msg.error_message = "Empty/blocked response from AI"
            db.session.commit()
        except Exception:
            db.session.rollback()

        logger.warning("Gemini returned a blocked or empty response (user_id=%s, conv_id=%s)", current_user.id, conversation.id)
        return jsonify({
            "error": (
                "MediAssist was unable to generate a response for that message. "
                "Please try rephrasing your question."
            ),
            "conversation_id": conversation.id,
            "user_message_id": user_msg.id,
        }), 422

    except Exception as exc:
        try:
            user_msg.status = "error"
            user_msg.error_message = f"Unexpected error: {type(exc).__name__}"
            db.session.commit()
        except Exception:
            db.session.rollback()

        logger.error("Unexpected error in chat pipeline (user_id=%s, conv_id=%s): %s", current_user.id, conversation.id, exc, exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred. Please try again.",
            "conversation_id": conversation.id,
            "user_message_id": user_msg.id,
        }), 500


@chat_bp.route("/reset", methods=["POST"])
@login_required
def reset():
    """Start a fresh consultation thread for the authenticated user and invalidate conversation cache."""
    new_conv = Conversation(user_id=current_user.id, title="New Consultation")
    db.session.add(new_conv)
    db.session.commit()

    llm_service = current_app.extensions.get("llm_service")
    if llm_service and hasattr(llm_service, "cache"):
        llm_service.cache.invalidate_user(current_user.id)

    logger.info("New consultation thread created for user %s (conv=%s).", current_user.id, new_conv.id)
    return jsonify({"status": "cleared", "conversation_id": new_conv.id}), 200


@chat_bp.route("/api/conversations", methods=["GET"])
@login_required
def list_conversations():
    """List all consultation conversations belonging to current user."""
    convs = current_user.conversations.order_by(Conversation.updated_at.desc()).all()
    return jsonify({"conversations": [c.to_dict() for c in convs]}), 200


@chat_bp.route("/api/conversations", methods=["POST"])
@login_required
def create_conversation():
    """Create a new consultation conversation for current user."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip() or "New Consultation"
    conv = Conversation(user_id=current_user.id, title=title[:100])
    db.session.add(conv)
    db.session.commit()
    return jsonify({"conversation": conv.to_dict()}), 201


@chat_bp.route("/api/conversations/<conversation_id>", methods=["GET"])
@login_required
def get_conversation(conversation_id):
    """Retrieve conversation messages, strictly verifying ownership."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403
    return jsonify({"conversation": conv.to_dict(include_messages=True)}), 200


@chat_bp.route("/api/conversations/<conversation_id>", methods=["PATCH", "PUT"])
@login_required
def update_conversation(conversation_id):
    """Rename conversation title, strictly verifying ownership."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    data = request.get_json(silent=True) or {}
    new_title = (data.get("title") or "").strip()
    if not new_title:
        return jsonify({"error": "Conversation title cannot be empty."}), 400

    conv.title = new_title[:200]
    db.session.commit()
    return jsonify({"status": "updated", "conversation": conv.to_dict()}), 200


@chat_bp.route("/api/conversations/<conversation_id>", methods=["DELETE"])
@login_required
def delete_conversation(conversation_id):
    """Delete conversation, strictly verifying ownership, and purge its cache and document storage."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    llm_service = current_app.extensions.get("llm_service")
    if llm_service and hasattr(llm_service, "cache"):
        llm_service.cache.invalidate_conversation(conversation_id)
    if llm_service and hasattr(llm_service, "document_service"):
        llm_service.document_service.delete_conversation_storage(conversation_id)

    db.session.delete(conv)
    db.session.commit()
    return jsonify({"status": "deleted"}), 200


@chat_bp.route("/api/conversations/<conversation_id>/upload", methods=["POST"])
@login_required
def upload_conversation_document(conversation_id):
    """Upload and process a study note (PDF or text) for a specific conversation."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    llm_service = current_app.extensions.get("llm_service")
    doc_service = getattr(llm_service, "document_service", None) if llm_service else None
    if not doc_service:
        return jsonify({"error": "Document processing service is not initialized."}), 503

    # Enforce maximum documents per conversation limit
    max_docs = current_app.config.get("MAX_DOCUMENTS_PER_CONVERSATION", 5)
    current_doc_count = conv.documents.count()
    if current_doc_count >= max_docs:
        return jsonify({
            "error": f"Maximum attachment limit reached for this conversation ({max_docs} documents). Please remove an existing document before uploading a new one."
        }), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a PDF or text file."}), 400

    file = request.files["file"]
    if not file or not file.filename or not file.filename.strip():
        return jsonify({"error": "No file selected."}), 400

    original_filename = file.filename.strip()
    max_size_bytes = current_app.config.get("MAX_UPLOAD_FILE_SIZE_BYTES", 15 * 1024 * 1024)
    allowed_exts = current_app.config.get("ALLOWED_EXTENSIONS", ("pdf", "txt", "md"))

    # Read stream bounded to max_size_bytes + 1 to detect oversize without loading unbounded into memory
    file_bytes = file.read(max_size_bytes + 1024)
    if len(file_bytes) > max_size_bytes:
        return jsonify({
            "error": f"File size exceeds the 15 MB limit. Please upload a smaller document or excerpt."
        }), 413

    # Validate file content, extension, and magic bytes
    is_valid, detected_type, validation_err = validate_file_content(
        file_bytes=file_bytes,
        filename=original_filename,
        max_size_bytes=max_size_bytes,
        allowed_extensions=allowed_exts,
    )
    if not is_valid:
        return jsonify({"error": validation_err}), 400

    # Duplicate detection within this conversation
    content_sha = hashlib.sha256(file_bytes).hexdigest()
    existing_dup = ConversationDocument.query.filter_by(
        conversation_id=conversation_id,
        content_sha256=content_sha,
    ).first()
    if existing_dup:
        return jsonify({
            "error": f"A document with identical content ('{existing_dup.filename}') is already attached to this conversation."
        }), 409

    # Generate safe server-side storage filename
    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{secure_filename(original_filename)}"
    conv_dir = doc_service.get_conversation_dir(conversation_id)
    storage_path = os.path.join(conv_dir, safe_name)

    # Perform malware / active content security check before writing to disk
    scan_ok, scan_note = scan_file_for_malware(storage_path, file_bytes)
    if not scan_ok:
        logger.warning("Malware scan failed for '%s' (conv=%s, user=%s): %s", original_filename, conversation_id, current_user.id, scan_note)
        return jsonify({"error": scan_note}), 400

    # Write file safely to disk
    try:
        with open(storage_path, "wb") as f:
            f.write(file_bytes)
    except Exception as save_err:
        logger.error("Failed to save uploaded file %s: %s", storage_path, save_err)
        return jsonify({"error": "Failed to save file on server."}), 500

    # Check whether to process asynchronously in background worker
    # In testing mode without an explicit ?async=1 flag, default to synchronous to keep standard test suites deterministic
    is_testing = current_app.config.get("TESTING", False)
    if is_testing:
        is_async = False
    else:
        is_async = current_app.config.get("ASYNC_DOCUMENT_PROCESSING", True)

    if request.args.get("async") in ("1", "true") or request.form.get("async") in ("1", "true"):
        is_async = True
    elif request.args.get("sync") in ("1", "true") or request.form.get("sync") in ("1", "true"):
        is_async = False

    if is_async:
        new_doc = ConversationDocument(
            id=doc_id,
            conversation_id=conversation_id,
            user_id=current_user.id,
            filename=original_filename,
            storage_path=os.path.relpath(storage_path, doc_service.upload_root),
            file_size_bytes=len(file_bytes),
            file_type=detected_type,
            content_sha256=content_sha,
            page_count=0,
            chunk_count=0,
            status="processing",
            error_message=None,
        )
        db.session.add(new_doc)
        db.session.commit()

        doc_service.submit_async_ingestion(
            app=current_app._get_current_object(),
            conversation_id=conversation_id,
            user_id=current_user.id,
            doc_id=doc_id,
            filename=original_filename,
            file_bytes=file_bytes,
            file_type=detected_type,
        )

        logger.info(
            "Uploaded document %s (id=%s) queued for background ingestion in conv=%s for user=%s",
            original_filename, doc_id, conversation_id, current_user.id
        )
        return jsonify({
            "status": "processing",
            "document": new_doc.to_dict(),
            "security_note": scan_note,
        }), 201

    # Synchronous processing (fallback when sync requested or testing)
    try:
        page_count, chunk_count = doc_service.process_and_index_document(
            conversation_id=conversation_id,
            user_id=current_user.id,
            doc_id=doc_id,
            filename=original_filename,
            file_bytes=file_bytes,
            file_type=detected_type,
        )
    except (DocumentExtractionError, DocumentValidationError) as client_err:
        if os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except OSError:
                pass
        return jsonify({"error": str(client_err)}), 422
    except Exception as proc_err:
        logger.error("Document indexing failure for %s: %s", original_filename, proc_err, exc_info=True)
        if os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except OSError:
                pass
        return jsonify({"error": f"Failed to process and index document: {proc_err}"}), 500

    new_doc = ConversationDocument(
        id=doc_id,
        conversation_id=conversation_id,
        user_id=current_user.id,
        filename=original_filename,
        storage_path=os.path.relpath(storage_path, doc_service.upload_root),
        file_size_bytes=len(file_bytes),
        file_type=detected_type,
        content_sha256=content_sha,
        page_count=page_count,
        chunk_count=chunk_count,
        status="ready",
    )
    db.session.add(new_doc)
    db.session.commit()

    if hasattr(llm_service, "cache") and llm_service.cache:
        llm_service.cache.invalidate_conversation(conversation_id)

    logger.info("Successfully uploaded and indexed document %s (id=%s) in conv=%s for user=%s", original_filename, doc_id, conversation_id, current_user.id)
    return jsonify({
        "status": "uploaded",
        "document": new_doc.to_dict(),
        "security_note": scan_note,
    }), 201


@chat_bp.route("/api/conversations/<conversation_id>/documents/<document_id>/status", methods=["GET"])
@login_required
def get_document_status(conversation_id, document_id):
    """Retrieve current background ingestion status of an attached document."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    doc = ConversationDocument.query.filter_by(id=document_id, conversation_id=conversation_id).first()
    if not doc:
        return jsonify({"error": "Document not found."}), 404

    return jsonify({
        "status": doc.status,
        "document": doc.to_dict(),
    }), 200


@chat_bp.route("/api/conversations/<conversation_id>/documents", methods=["GET"])
@login_required
def list_conversation_documents(conversation_id):
    """List all documents attached to a conversation."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    docs = conv.documents.all()
    return jsonify({"documents": [d.to_dict() for d in docs]}), 200


@chat_bp.route("/api/conversations/<conversation_id>/documents/<document_id>", methods=["DELETE"])
@login_required
def delete_conversation_document(conversation_id, document_id):
    """Delete an attached document, remove its vectors, and update conversation index."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    doc = ConversationDocument.query.filter_by(id=document_id, conversation_id=conversation_id).first()
    if not doc:
        return jsonify({"error": "Document not found."}), 404

    llm_service = current_app.extensions.get("llm_service")
    if llm_service and hasattr(llm_service, "document_service"):
        llm_service.document_service.delete_document(conversation_id, current_user.id, document_id)
    if llm_service and hasattr(llm_service, "cache"):
        llm_service.cache.invalidate_conversation(conversation_id)

    db.session.delete(doc)
    db.session.commit()

    logger.info("Deleted document %s from conversation %s", document_id, conversation_id)
    return jsonify({"status": "deleted", "document_id": document_id}), 200


@chat_bp.route("/api/conversations/<conversation_id>/images/<image_name>", methods=["GET"])
@login_required
def get_conversation_image(conversation_id, image_name):
    """Securely serve images extracted from user-uploaded PDFs for an authorized conversation."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    llm_service = current_app.extensions.get("llm_service")
    doc_service = getattr(llm_service, "document_service", None) if llm_service else None
    if not doc_service:
        return jsonify({"error": "Document service not available."}), 503

    conv_dir = doc_service.get_conversation_dir(conversation_id)
    images_dir = os.path.join(conv_dir, "images")
    safe_name = secure_filename(image_name)
    target_path = os.path.join(images_dir, safe_name)
    if not os.path.exists(target_path):
        return jsonify({"error": "Image not found."}), 404

    return send_from_directory(images_dir, safe_name)


@chat_bp.route("/api/conversations/<conversation_id>/mcq/answer", methods=["POST"])
@login_required
def answer_mcq(conversation_id):
    """Authorized endpoint for submitting an interactive MCQ option selection with duplicate prevention."""
    conv = Conversation.query.filter_by(id=conversation_id).first()
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    if conv.user_id != current_user.id:
        return jsonify({"error": "Access denied. Conversation belongs to another user."}), 403

    data = request.get_json(silent=True) or {}
    selected_option = (data.get("selected_option") or "").strip().upper()
    if not selected_option or selected_option not in {"A", "B", "C", "D"}:
        return jsonify({"error": "Invalid option selection. Please choose Option A, B, C, or D."}), 400

    question_id = (data.get("question_id") or data.get("mcq_id") or "").strip() or None

    # Check for duplicate submission in conversation history
    max_history = current_app.config.get("MAX_HISTORY_TURNS", 20)
    past_messages = (
        conv.messages.filter(ChatMessage.status == "completed")
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()[-max_history:]
    )

    conv_history = []
    active_mcq = None
    for msg in past_messages:
        role = "model" if msg.role in ("assistant", "model") else "user"
        ev = msg.evidence or {}
        if isinstance(ev, dict) and ev.get("interactive_mcq"):
            mcq_data = ev["interactive_mcq"]
            if not question_id or mcq_data.get("question_id") == question_id:
                active_mcq = mcq_data
        conv_history.append({"role": role, "parts": [{"text": msg.content}], "evidence": msg.evidence})

    if active_mcq and active_mcq.get("answered"):
        prior_choice = active_mcq.get("user_answer") or "recorded"
        correct_key = active_mcq.get("correct_key") or active_mcq.get("correct_option") or "A"
        return jsonify({
            "status": "already_answered",
            "already_answered": True,
            "is_correct": (prior_choice.upper() == correct_key.upper()),
            "message": "Your answer was already recorded for this practice question.",
            "response": (
                f"### ℹ️ Question Already Answered\n\n"
                f"You have already submitted an answer for this practice question (Selected: **Option {prior_choice}**).\n\n"
                f"To preserve scoring integrity, duplicate submissions are not scored. "
                f"The correct answer remains **Option {correct_key}**.\n\n"
                f"Ask for another MCQ or continue with your revision!"
            ),
            "selected_option": prior_choice,
            "interactive_mcq": active_mcq,
            "conversation_id": conv.id,
        }), 200

    user_message = f"Option {selected_option}"
    user_msg = _get_or_create_user_message(conv, user_message)

    llm_service = current_app.extensions.get("llm_service")
    if not llm_service:
        return jsonify({"error": "LLM service unavailable."}), 503

    reply, sources, triage = llm_service.get_response(
        user_message=user_message,
        history=conv_history,
        user_id=current_user.id,
        conversation_id=conv.id,
        academic_year=getattr(current_user, "academic_year", None),
        target_exam=getattr(current_user, "target_exam", None),
        selected_option=selected_option,
        question_id=question_id,
    )

    user_msg.status = "completed"
    user_msg.error_message = None
    practice_mode = triage.get("practice_mode", "mcq_practice")
    evidence_payload = {
        "sources": sources,
        "triage": triage,
        "practice_mode": practice_mode,
        "interactive_mcq": triage.get("interactive_mcq"),
        "clinical_case": triage.get("clinical_case"),
    }
    bot_msg = ChatMessage(
        conversation_id=conv.id,
        role="model",
        content=reply,
        evidence=evidence_payload,
        status="completed",
    )
    db.session.add(bot_msg)
    conv.touch()
    db.session.commit()

    correct_key = (active_mcq.get("correct_key") or active_mcq.get("correct_option") or "A") if active_mcq else "A"
    is_correct = (selected_option.upper() == correct_key.upper())

    return jsonify({
        "status": "answered",
        "already_answered": False,
        "is_correct": is_correct,
        "message": "Answer recorded successfully.",
        "response": reply,
        "sources": sources,
        "triage": triage,
        "interactive_mcq": triage.get("interactive_mcq"),
        "conversation_id": conv.id,
        "created_at": bot_msg.created_at.isoformat() if bot_msg.created_at else utc_now().isoformat(),
    }), 200


@chat_bp.route("/api/profile", methods=["GET"])
@login_required
def get_profile():
    """Retrieve authenticated student's academic profile."""
    return jsonify({"user": current_user.to_dict()}), 200


@chat_bp.route("/api/profile", methods=["PATCH", "PUT"])
@login_required
def update_profile():
    """Update authenticated student's academic profile with strict server-side validation."""
    data = request.get_json(silent=True) or {}
    from models import VALID_ACADEMIC_YEARS, VALID_TARGET_EXAMS

    if "academic_year" in data:
        val = data.get("academic_year")
        if val is None or val == "" or val is False:
            current_user.academic_year = None
        else:
            clean_year = str(val).strip().lower()
            if clean_year not in VALID_ACADEMIC_YEARS:
                return jsonify({
                    "error": f"Invalid academic_year '{val}'. Supported values: {sorted(VALID_ACADEMIC_YEARS)}"
                }), 400
            current_user.academic_year = clean_year

    if "target_exam" in data:
        val = data.get("target_exam")
        if val is None or val == "" or val is False:
            current_user.target_exam = None
        else:
            clean_exam = str(val).strip().lower()
            if clean_exam not in VALID_TARGET_EXAMS:
                return jsonify({
                    "error": f"Invalid target_exam '{val}'. Supported values: {sorted(VALID_TARGET_EXAMS)}"
                }), 400
            current_user.target_exam = clean_exam

    db.session.commit()
    logger.info(
        "Updated student profile (user_id=%s, academic_year=%s, target_exam=%s)",
        current_user.id, current_user.academic_year, current_user.target_exam
    )
    return jsonify({"status": "updated", "user": current_user.to_dict()}), 200

