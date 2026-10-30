"""Core Application Routes.

Provides public landing page, static build assets, health check, and document metadata.
"""

from __future__ import annotations

import json
import os
from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required

from models import Conversation, db

core_bp = Blueprint("core", __name__)


@core_bp.route("/")
def index():
    """Public landing page for unauthenticated visitors; redirects authenticated users to consultation."""
    if current_user.is_authenticated:
        return redirect(url_for("chat.chat_view"))
    return render_template("landing.html")


@core_bp.route("/dashboard")
def dashboard():
    """Obsolete dashboard route: safely redirect to consultation interface."""
    if current_user.is_authenticated:
        return redirect(url_for("chat.chat_view"))
    return redirect(url_for("auth.login", next=url_for("chat.chat_view")))


@core_bp.route("/assets/<path:filename>")
def serve_dist_assets(filename):
    """Serve built frontend assets (CSS, JS) from static/dist/assets."""
    dist_assets = os.path.join(current_app.root_path, "static", "dist", "assets")
    return send_from_directory(dist_assets, filename)


@core_bp.route("/api/status", methods=["GET"])
@login_required
def api_status():
    """System health, active model, and knowledge-base status."""
    llm_service = current_app.extensions.get("llm_service")
    retriever = getattr(llm_service, "retriever", None)
    model_name = current_app.config.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    chunks_count = retriever.index.ntotal if retriever and retriever.is_ready() else 0

    return jsonify({
        "status": "healthy",
        "model": model_name,
        "retrieval_mode": retriever.retrieval_mode if retriever else "hybrid",
        "knowledge_chunks": chunks_count,
    })


@core_bp.route("/api/documents", methods=["GET"])
@login_required
def get_documents():
    """Protected medical document retrieval metadata."""
    meta_path = os.path.join(current_app.root_path, "knowledge", "chunks_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            docs = []
            for item in metadata:
                docs.append({
                    "chunk_id": item.get("chunk_id"),
                    "passage_id": item.get("passage_id"),
                    "document_id": item.get("document_id"),
                    "source_title": item.get("source_title"),
                    "source_url": item.get("source_url"),
                    "section": item.get("section"),
                })
            return jsonify({"count": len(docs), "documents": docs}), 200
        except Exception:
            return jsonify({"count": 0, "documents": []}), 200
    return jsonify({"count": 0, "documents": []}), 200
