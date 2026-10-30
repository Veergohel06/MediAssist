"""Comprehensive verification script for UI/UX redesign and Dashboard removal."""

import re
import requests

BASE_URL = "http://127.0.0.1:5000"


def run_checks():
    print("==================================================")
    print("  MediAssist UI/UX & Flow Verification Suite")
    print("==================================================")
    session = requests.Session()

    # 1. Open application without login -> Public landing page
    r = session.get(f"{BASE_URL}/", allow_redirects=False)
    assert r.status_code == 200, f"Expected 200 on /, got {r.status_code}"
    assert "Evidence-Grounded Healthcare AI" in r.text
    assert "Sign In" in r.text
    assert "Dashboard" not in r.text
    print("[PASS] Test 1: Unauthenticated / serves Public Landing Page (no Dashboard link)")

    # 2. Open old dashboard URL when unauthenticated -> Safe redirect to login
    r = session.get(f"{BASE_URL}/dashboard", allow_redirects=False)
    assert r.status_code == 302, f"Expected 302 on unauthenticated /dashboard, got {r.status_code}"
    assert "/login" in r.headers.get("Location", "")
    print(f"[PASS] Test 2: Unauthenticated /dashboard safely redirects to {r.headers.get('Location')}")

    # 3. Unauthenticated /chat -> redirects to login
    r = session.get(f"{BASE_URL}/chat", allow_redirects=False)
    assert r.status_code == 302
    assert "/login" in r.headers.get("Location", "")
    print(f"[PASS] Test 3: Unauthenticated /chat safely redirects to {r.headers.get('Location')}")

    # 4. Inspect Login Page styling & elements
    r = session.get(f"{BASE_URL}/login")
    assert r.status_code == 200
    assert 'class="form-control"' in r.text
    assert 'class="password-input-wrapper"' in r.text
    assert 'class="password-toggle-btn"' in r.text
    assert 'class="form-checkbox"' in r.text
    assert "👁️" not in r.text, "Emoji eye should be replaced with SVG"
    print("[PASS] Test 4: Login page has clean form-control, SVG toggle, and themed checkbox")

    # Extract CSRF token
    csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', r.text)
    csrf_token = csrf_match.group(1) if csrf_match else ""
    assert csrf_token, "Could not extract CSRF token"

    # 5. Perform Login with existing or new user
    login_data = {
        "csrf_token": csrf_token,
        "email": "doctor_test_ui@hospital.org",
        "password": "Password123!",
    }
    # Register first if needed
    reg_r = session.post(
        f"{BASE_URL}/register",
        data={
            "csrf_token": csrf_token,
            "name": "Dr. Christopher Alexander Montgomery-Smith III",
            "email": "doctor_test_ui@hospital.org",
            "password": "Password123!",
            "password_confirm": "Password123!",
        },
        allow_redirects=False,
    )
    if reg_r.status_code == 302:
        print(f"[PASS] Test 5a: Registration successfully redirected to {reg_r.headers.get('Location')}")
        assert "/chat" in reg_r.headers.get("Location", "")
    else:
        # In case user already exists, login:
        login_r = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=False)
        assert login_r.status_code == 302, f"Expected 302 from /login, got {login_r.status_code}: {login_r.text[:200]}"
        redirect_loc = login_r.headers.get("Location", "")
        assert "/chat" in redirect_loc, f"Expected redirect to /chat, got {redirect_loc}"
        print(f"[PASS] Test 5b: Login successfully redirects directly to {redirect_loc} (NO dashboard)")

    # 6. Authenticated user opening / -> redirects to /chat
    auth_root = session.get(f"{BASE_URL}/", allow_redirects=False)
    assert auth_root.status_code == 302
    assert "/chat" in auth_root.headers.get("Location", "")
    print(f"[PASS] Test 6: Authenticated user accessing / redirects directly to {auth_root.headers.get('Location')}")

    # 7. Authenticated user opening legacy /dashboard -> redirects to /chat
    auth_dash = session.get(f"{BASE_URL}/dashboard", allow_redirects=False)
    assert auth_dash.status_code == 302
    assert "/chat" in auth_dash.headers.get("Location", "")
    print(f"[PASS] Test 7: Authenticated user accessing legacy /dashboard safely redirects to {auth_dash.headers.get('Location')}")

    # 8. Inspect Consultation Interface (/chat)
    chat_r = session.get(f"{BASE_URL}/chat")
    assert chat_r.status_code == 200
    chat_html = chat_r.text
    assert "Dashboard" not in chat_html, "Dashboard button must not be present anywhere in chat.html"
    assert "MBBS Level" not in chat_html, "MBBS Level selector must not be present in chat.html"
    assert "academic-profile-btn" not in chat_html, "academic profile button must not be present"
    assert "nav-profile-container" in chat_html
    assert "currentUserDisplay" in chat_html
    assert "Dr. Christopher Alexander Montgomery-Smith III" in chat_html
    assert "Sign Out" in chat_html
    print("[PASS] Test 8: Consultation interface is clean: NO Dashboard button, NO MBBS Level selector, single-line username with compact profile dropdown")

    # 9. Test real AI chat message endpoint /chat
    chat_post = session.post(
        f"{BASE_URL}/chat",
        headers={"X-CSRFToken": csrf_token},
        json={"message": "What are common signs of iron deficiency anemia?"},
    )
    assert chat_post.status_code == 200, f"Expected 200 from /chat, got {chat_post.status_code}"
    res_data = chat_post.json()
    assert "response" in res_data or "reply" in res_data or "content" in res_data
    print(f"[PASS] Test 9: Real Chatbot backend responded successfully (length: {len(res_data.get('response', ''))} chars)")

    # 10. Perform Logout -> Redirect to / landing page
    logout_r = session.post(f"{BASE_URL}/logout", data={"csrf_token": csrf_token}, allow_redirects=False)
    assert logout_r.status_code == 302
    assert logout_r.headers.get("Location", "") == "/"
    print("[PASS] Test 10: Logout cleared session and redirected to public landing page (/)")

    # 11. Post-logout /chat access is blocked
    after_logout = session.get(f"{BASE_URL}/chat", allow_redirects=False)
    assert after_logout.status_code == 302
    assert "/login" in after_logout.headers.get("Location", "")
    print("[PASS] Test 11: Post-logout access to /chat is securely rejected to /login")

    print("\n==================================================")
    print("  ALL 11 END-TO-END VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_checks()
