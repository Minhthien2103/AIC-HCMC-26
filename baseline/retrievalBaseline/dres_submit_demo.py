# dres_client.py
# pip install requests

import requests
from typing import List, Optional, Dict, Any

# ==============================
# 🔧 CẤU HÌNH — ĐIỀN Ở ĐÂY
# ==============================
DRES_BASE_URL = "http://<your-dres-host>:5000"   # VD: "http://192.168.28.151:5000"
SESSION_ID: Optional[str] = None                  # Điền thủ công nếu bạn có sẵn, VD: "abcd-efgh-1234"
USERNAME: Optional[str] = "<username>"            # Dùng khi cần auto-login
PASSWORD: Optional[str] = "<password>"            # Dùng khi cần auto-login
DEFAULT_FPS: float = 25.0                         # FPS mặc định để đổi frame -> milliseconds

# ==============================
# 🧩 KIỂU DỮ LIỆU TỐI THIỂU
# ==============================
# Tối giản theo code JS: ResultItem có ít nhất videoId (str) và timestamp (frame index dạng str/int)
ResultItem = Dict[str, Any]


# =========================================
# 🔐 Lấy session: thủ công hoặc auto-login
# =========================================
def get_session_id() -> str:
    """
    - Nếu biến SESSION_ID đã được điền thủ công -> dùng luôn.
    - Ngược lại, thử gọi POST /api/v2/login với USERNAME/PASSWORD để nhận sessionId.
    """
    if SESSION_ID:
        print("[info] Using manual SESSION_ID.")
        return SESSION_ID

    if not USERNAME or not PASSWORD:
        raise RuntimeError(
            "No SESSION_ID provided and USERNAME/PASSWORD not set. "
            "Either fill SESSION_ID OR provide USERNAME/PASSWORD for auto-login."
        )

    login_url = f"{DRES_BASE_URL}/api/v2/login"
    payload = {"username": USERNAME, "password": PASSWORD}
    resp = requests.post(login_url, json=payload, timeout=10)
    if not resp.ok:
        try:
            err = resp.json()
        except Exception:
            err = {"error": resp.text}
        raise RuntimeError(f"Login failed: HTTP {resp.status_code} - {err}")

    data = resp.json()
    sid = data.get("sessionId") or data.get("sessionID") or data.get("session_id")
    if not sid:
        raise RuntimeError(f"Login response missing sessionId: {data}")
    print("[info] Auto-login success. sessionId =", sid)
    return sid


# =========================================
# 🧭 Lấy evaluation đang ACTIVE
# =========================================
def get_active_evaluation_id(session_id: str) -> str:
    url = f"{DRES_BASE_URL}/api/v2/client/evaluation/list"
    resp = requests.get(url, params={"session": session_id}, timeout=10)
    if not resp.ok:
        raise RuntimeError(f"Get evaluation list failed: HTTP {resp.status_code} - {resp.text}")

    evaluations = resp.json()
    active = next((e for e in evaluations if str(e.get("status")).upper() == "ACTIVE"), None)
    if not active:
        raise RuntimeError("No active evaluation found.")
    return str(active.get("id"))


# ======================================================
# ⏱️ Chuyển frame index -> milliseconds với DEFAULT_FPS
# ======================================================
def ms_from_frame_index(frame_value: Any, fps: float = DEFAULT_FPS) -> int:
    """
    frame_value: có thể là str hoặc int; ví dụ '123' -> 123
    fps: frames/second (mặc định DEFAULT_FPS)
    return: milliseconds (int)
    """
    frame_index = int(frame_value)
    return int((frame_index / fps) * 1000)




# =========================================
# 📤 Submit 1 kết quả (điểm thời gian) hoặc VQA
# =========================================
def submit_result(
    result: ResultItem,
    session_id: str,
    evaluation_id: str,
    question: Optional[str] = None,
    fps: float = DEFAULT_FPS,
) -> Dict[str, Any]:
    """
    - Nếu có 'question' => submit VQA (text).
    - Nếu không => submit dạng mediaItemName + start/end (ms).
    """
    video_id = str(result["videoId"])
    timestamp = result["timestamp"]  # frame index

    if question:
        # VQA
        ms = ms_from_frame_index(timestamp, fps=fps)
        text = f"QA-{question}-{video_id}-{ms}"
        body = {"answerSets": [{"answers": [{"text": text}]}]}
    else:
        # Standard temporal answer (điểm thời gian)
        ms = ms_from_frame_index(timestamp, fps=fps)
        body = {
            "answerSets": [
                {
                    "answers": [
                        {"mediaItemName": video_id, "start": ms, "end": ms}
                    ]
                }
            ]
        }

    url = f"{DRES_BASE_URL}/api/v2/submit/{evaluation_id}"
    resp = requests.post(url, params={"session": session_id}, json=body, timeout=15)
    if not resp.ok:
        try:
            raise RuntimeError(resp.json().get("description", resp.text))
        except Exception:
            raise RuntimeError(f"DRES submission failed: HTTP {resp.status_code} - {resp.text}")

    data = resp.json()
    print("[ok] DRES submission:", data)
    return data





def full_submission_flow(result: ResultItem, question: Optional[str] = None, fps: float = DEFAULT_FPS) -> Dict[str, Any]:
    """
    1) Lấy sessionId (thủ công hoặc auto-login)
    2) Lấy evaluation ACTIVE
    3) Gửi submit 1 kết quả hoặc VQA
    """
    session_id = get_session_id()
    evaluation_id = get_active_evaluation_id(session_id)
    return submit_result(result, session_id, evaluation_id, question=question, fps=fps)


# =========================================
# 🧪 Ví dụ dùng thử (chạy trực tiếp file)
# =========================================
if __name__ == "__main__":
    # 🔹 CÁCH 1: điền SESSION_ID sẵn ở trên -> giữ USERNAME/PASSWORD như cũ cũng được
    # 🔹 CÁCH 2: để SESSION_ID=None -> điền USERNAME/PASSWORD để auto-login

    # Ví dụ submit 1 điểm thời gian
    sample_result = {"videoId": "K01_V001", "timestamp": "123"}  # frame index = 123
    try:
        resp1 = full_submission_flow(sample_result)  # không có question => standard
        print("Submit standard OK:", resp1)
    except Exception as e:
        print("Submit standard error:", e)


