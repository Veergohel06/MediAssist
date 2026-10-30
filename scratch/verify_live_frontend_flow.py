import re
import requests

BASE_URL = "http://127.0.0.1:5000"

def test_live_frontend_api_flow():
    session = requests.Session()
    
    # 1. Access login page and get CSRF token
    r_login_page = session.get(f"{BASE_URL}/login")
    assert r_login_page.status_code == 200, f"Expected 200, got {r_login_page.status_code}"
    
    csrf_match = re.search(r'name="csrf_token"\s+value="([^"]+)"', r_login_page.text)
    assert csrf_match, "CSRF token not found in login page"
    csrf_token = csrf_match.group(1)
    
    # 2. Login as live_test_user
    login_data = {
        "email": "live_test_user@example.com",
        "password": "Password123!",
        "csrf_token": csrf_token,
    }
    r_login = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=True)
    assert r_login.status_code == 200
    print("Logged in successfully as live_test_user@example.com")
    
    # 3. Access /chat page
    r_chat = session.get(f"{BASE_URL}/chat")
    assert r_chat.status_code == 200
    assert "conversationsList" in r_chat.text
    assert "newChatBtn" in r_chat.text
    assert "historyCountBadge" in r_chat.text
    print("Chat page loaded with conversationsList, newChatBtn, and historyCountBadge.")
    
    # Extract chat CSRF token
    meta_csrf = re.search(r'<meta\s+name="csrf-token"\s+content="([^"]+)"', r_chat.text).group(1)
    headers = {
        "X-CSRFToken": meta_csrf,
        "Content-Type": "application/json"
    }
    
    # 4. List conversations
    r_convs = session.get(f"{BASE_URL}/api/conversations")
    assert r_convs.status_code == 200
    convs_data = r_convs.json()
    print(f"Initial conversation count: {len(convs_data.get('conversations', []))}")
    
    # 5. Create / Start a consultation turn via POST /chat
    r_chat_post = session.post(f"{BASE_URL}/chat", headers=headers, json={
        "message": "What are common signs of tension headaches?"
    })
    assert r_chat_post.status_code == 200
    chat_resp = r_chat_post.json()
    conv_id = chat_resp["conversation_id"]
    print(f"Created conversation ID: {conv_id}")
    assert "response" in chat_resp
    assert "created_at" in chat_resp
    
    # 6. Fetch conversation details (verifying messages, sources, timestamps)
    r_detail = session.get(f"{BASE_URL}/api/conversations/{conv_id}")
    assert r_detail.status_code == 200
    detail = r_detail.json()["conversation"]
    assert detail["id"] == conv_id
    assert len(detail["messages"]) >= 2
    user_turn = detail["messages"][0]
    bot_turn = detail["messages"][1]
    assert user_turn["role"] == "user"
    assert bot_turn["role"] == "model"
    assert "created_at" in bot_turn
    sources = bot_turn.get("evidence", {}).get("sources", [])
    print(f"Messages restored: {len(detail['messages'])}, verified sources count: {len(sources)}")
    
    # 7. Rename conversation via PATCH /api/conversations/<id>
    new_title = "Live Renamed Headache Guidance"
    r_rename = session.patch(f"{BASE_URL}/api/conversations/{conv_id}", headers=headers, json={
        "title": new_title
    })
    assert r_rename.status_code == 200
    renamed_conv = r_rename.json()["conversation"]
    assert renamed_conv["title"] == new_title
    print(f"Renamed successfully to: '{renamed_conv['title']}'")
    
    # Verify via list endpoint
    r_convs2 = session.get(f"{BASE_URL}/api/conversations")
    assert r_convs2.status_code == 200
    c_list = r_convs2.json()["conversations"]
    assert any(c["id"] == conv_id and c["title"] == new_title for c in c_list)
    print("Verified renamed conversation appears in conversation list.")
    
    # 8. Reset conversation (New Consultation) via POST /reset
    r_reset = session.post(f"{BASE_URL}/reset", headers=headers)
    assert r_reset.status_code == 200
    print("New Consultation /reset endpoint responded 200 OK.")
    
    # 9. Delete conversation via DELETE /api/conversations/<id>
    r_del = session.delete(f"{BASE_URL}/api/conversations/{conv_id}", headers=headers)
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "deleted"
    print("Delete conversation responded 200 OK.")
    
    # Confirm 404 after delete
    r_after_del = session.get(f"{BASE_URL}/api/conversations/{conv_id}")
    assert r_after_del.status_code == 404
    print("Confirmed 404 after deletion.")

if __name__ == "__main__":
    test_live_frontend_api_flow()
    print("\n>>> ALL LIVE FRONTEND APIS VERIFIED PERFECTLY ON RUNNING FLASK SERVER! <<<")
