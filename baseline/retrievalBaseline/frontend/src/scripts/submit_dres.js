// ==============================================================================
// DRES Submission & Authentication Manager for AIC 2026
// Conforms to VBS / DRES v2.0 & AI Challenge 2026 Official Rules
// ==============================================================================

let isSubmitting = false;

function showTemporaryAlert(message, type = 'info') {
    const alertElement = document.createElement('div');
    alertElement.textContent = message;
    alertElement.style.position = 'fixed';
    alertElement.style.top = '25px';
    alertElement.style.left = '50%';
    alertElement.style.transform = 'translateX(-50%)';
    alertElement.style.padding = '12px 24px';
    alertElement.style.borderRadius = '8px';
    alertElement.style.zIndex = '999999';
    alertElement.style.fontSize = '14px';
    alertElement.style.fontWeight = '600';
    alertElement.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.6)';
    alertElement.style.transition = 'all 0.3s ease';

    if (type === 'success') {
        alertElement.style.backgroundColor = '#059669';
        alertElement.style.color = '#ffffff';
        alertElement.style.border = '1px solid #34d399';
    } else if (type === 'error') {
        alertElement.style.backgroundColor = '#dc2626';
        alertElement.style.color = '#ffffff';
        alertElement.style.border = '1px solid #f87171';
    } else if (type === 'warning') {
        alertElement.style.backgroundColor = '#d97706';
        alertElement.style.color = '#ffffff';
        alertElement.style.border = '1px solid #fbbf24';
    } else {
        alertElement.style.backgroundColor = '#1e293b';
        alertElement.style.color = '#f8fafc';
        alertElement.style.border = '1px solid #38bdf8';
    }

    document.body.appendChild(alertElement);

    setTimeout(() => {
        alertElement.style.opacity = '0';
        setTimeout(() => alertElement.remove(), 300);
    }, 4000);
}

// ------------------------------------------------------------------------------
// DRES Configuration Helpers
// ------------------------------------------------------------------------------

function getDresBaseUrl() {
    return localStorage.getItem('dresHost') || 
           (window.CONFIG && window.CONFIG.DRES_BASE) || 
           "https://aiclub.uit.edu.vn/dres";
}

function getDresEvaluationId() {
    return localStorage.getItem('evaluationID') || 
           (window.CONFIG && window.CONFIG.DRES_EVALUATION_ID) || 
           "bcb482c6-2b9e-4908-877a-e5ccd7c04640";
}

function getDresSessionId() {
    return localStorage.getItem('contestSessionID') || 
           (window.CONFIG && window.CONFIG.DRES_SESSION_ID) || 
           "q-ekxzxUEkkxlpMwfkdcCvqt4CONUS6P";
}

function updateDresStatusBadge() {
    const curSession = getDresSessionId();
    const curUser = localStorage.getItem('userName') || '';
    const btnLabel = document.getElementById('dres-btn-label');
    const configBtn = document.getElementById('dres-config-btn');

    if (btnLabel && configBtn) {
        if (curSession && curSession.trim() !== '' && curSession !== 'None') {
            btnLabel.textContent = curUser ? `DRES (${curUser})` : `DRES (Connected)`;
            configBtn.classList.add('connected');
            configBtn.title = `DRES Đang kết nối | Session: ${curSession.substring(0, 8)}...`;
        } else {
            btnLabel.textContent = `DRES Login`;
            configBtn.classList.remove('connected');
            configBtn.title = `Chưa đăng nhập DRES - Click để cấu hình`;
        }
    }
}

window.setDresPreset = function(url) {
    const hostInput = document.getElementById('dres-host-input');
    if (hostInput) {
        hostInput.value = url;
    }
};

// ------------------------------------------------------------------------------
// DRES Modal UI Management
// ------------------------------------------------------------------------------

function openDresModal() {
    const modal = document.getElementById('dresModal');
    if (!modal) return;

    const hostInput = document.getElementById('dres-host-input');
    const evalInput = document.getElementById('dres-eval-input');
    const sessionInput = document.getElementById('dres-session-input');
    const statusDisplay = document.getElementById('dres-current-status');

    if (hostInput) hostInput.value = getDresBaseUrl();
    if (evalInput) evalInput.value = getDresEvaluationId();
    
    const curSession = getDresSessionId();
    if (sessionInput) sessionInput.value = curSession;

    if (statusDisplay) {
        if (curSession && curSession.trim() !== '' && curSession !== 'None') {
            statusDisplay.innerHTML = `<span style="color: #4ade80;">🟢 Đã kết nối Session: <code>${curSession.substring(0, 16)}...</code></span>`;
        } else {
            statusDisplay.innerHTML = `<span style="color: #f87171;">🔴 Chưa đăng nhập DRES / Chưa có Session ID</span>`;
        }
    }

    modal.style.display = 'flex';
}

function closeDresModal() {
    const modal = document.getElementById('dresModal');
    if (modal) modal.style.display = 'none';
    updateDresStatusBadge();
}

async function handleDresTestConnection() {
    const host = (document.getElementById('dres-host-input')?.value || getDresBaseUrl()).replace(/\/+$/, '');
    const testBtn = document.getElementById('dres-test-btn');
    const origText = testBtn ? testBtn.innerHTML : '';

    if (testBtn) {
        testBtn.disabled = true;
        testBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Testing...';
    }

    try {
        // Attempt lightweight reachability probe (DRES v2 client API)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3500);

        const res = await fetch(`${host}/api/v2/client/evaluation/list`, {
            method: 'GET',
            signal: controller.signal
        }).catch(async () => {
            // Fallback ping root
            return await fetch(`${host}/`, { method: 'GET', signal: controller.signal });
        });

        clearTimeout(timeoutId);
        showTemporaryAlert(`✅ Đã kết nối thành công tới ${host}! (HTTP ${res.status})`, "success");
        const statusDisplay = document.getElementById('dres-current-status');
        if (statusDisplay) {
            statusDisplay.innerHTML = `<span style="color: #38bdf8;">🌐 Máy chủ DRES hoạt động tốt tại <code>${host}</code></span>`;
        }
    } catch (err) {
        console.warn("[DRES Test Connection]", err);
        showTemporaryAlert(`⚠️ Không thể kết nối tới ${host}: ${err.message}. Kiểm tra lại VPN hoặc IP!`, "warning");
    } finally {
        if (testBtn) {
            testBtn.disabled = false;
            testBtn.innerHTML = origText;
        }
    }
}

async function handleDresLogin() {
    const host = (document.getElementById('dres-host-input')?.value || getDresBaseUrl()).replace(/\/+$/, '');
    const user = document.getElementById('dres-user-input')?.value?.trim();
    const pass = document.getElementById('dres-pass-input')?.value;
    const evalId = document.getElementById('dres-eval-input')?.value?.trim() || getDresEvaluationId();

    if (!user || !pass) {
        alert("Vui lòng nhập cả Tên đăng nhập và Mật khẩu của Đội thi!");
        return;
    }

    const btn = document.getElementById('dres-login-btn');
    const origText = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Đang đăng nhập DRES...';
    }

    try {
        const loginUrl = `${host}/api/v2/login`;
        console.log("[DRES] Logging in to:", loginUrl);

        const response = await fetch(loginUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                "username": user,
                "password": pass
            })
        });

        if (!response.ok) {
            let desc = `HTTP ${response.status}`;
            try {
                const errData = await response.json();
                desc = errData.description || errData.message || desc;
            } catch (e) {}
            throw new Error(`Đăng nhập thất bại (${response.status}): ${desc}`);
        }

        const data = await response.json();
        const sessionId = data.sessionId || data.sessionID || data.session_id;

        if (!sessionId) {
            throw new Error("Phản hồi từ DRES không chứa sessionId: " + JSON.stringify(data));
        }

        // Save credentials
        localStorage.setItem('dresHost', host);
        localStorage.setItem('userName', user);
        localStorage.setItem('contestSessionID', sessionId);
        localStorage.setItem('evaluationID', evalId);

        // Update display in modal
        const sessionInput = document.getElementById('dres-session-input');
        if (sessionInput) sessionInput.value = sessionId;
        const statusDisplay = document.getElementById('dres-current-status');
        if (statusDisplay) {
            statusDisplay.innerHTML = `<span style="color: #4ade80;">🟢 Đăng nhập thành công! Session: <code>${sessionId.substring(0, 16)}...</code></span>`;
        }

        showTemporaryAlert(`🎉 Đăng nhập DRES thành công (${user})!`, "success");
        updateDresStatusBadge();
        setTimeout(() => closeDresModal(), 1200);

    } catch (err) {
        console.error("[DRES Login Error]", err);
        alert(err.message);
        showTemporaryAlert(`❌ ${err.message}`, "error");
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = origText;
        }
    }
}

function handleDresSaveConfig() {
    const host = (document.getElementById('dres-host-input')?.value || getDresBaseUrl()).replace(/\/+$/, '');
    const evalId = document.getElementById('dres-eval-input')?.value?.trim() || getDresEvaluationId();
    const session = document.getElementById('dres-session-input')?.value?.trim() || '';

    localStorage.setItem('dresHost', host);
    localStorage.setItem('evaluationID', evalId);
    if (session) {
        localStorage.setItem('contestSessionID', session);
    }

    showTemporaryAlert("✅ Đã lưu cấu hình DRES!", "success");
    closeDresModal();
}

// ------------------------------------------------------------------------------
// Core Submission Function (submit_to_dres_v2)
// ------------------------------------------------------------------------------

async function submit_to_dres_v2() {
    if (isSubmitting) {
        showTemporaryAlert("⏳ Bài nộp trước đang được gửi, vui lòng chờ...", "warning");
        return;
    }

    // 1. Kiểm tra danh sách ảnh đã ghim trong Export Area
    if (typeof exportedImages === 'undefined' || !exportedImages || exportedImages.length === 0) {
        showTemporaryAlert("⚠️ Bạn chưa ghim frame nào! Hãy bấm '📌 Ghim' trên kết quả tìm kiếm trước.", "warning");
        alert("⚠️ BẠN CHƯA GHIM FRAME NÀO ĐỂ NỘP!\n\nCách thực hiện:\n1. Tìm kiếm và chọn kết quả video/frame ưng ý.\n2. Rê chuột vào ảnh và bấm chữ '📌 Ghim' để đưa frame vào khung Export Area (bên phải).\n3. Bấm lại nút 'Submit DRES' để nộp ngay.");
        return;
    }

    // 2. Kiểm tra Session ID & Evaluation ID
    const sessionId = getDresSessionId();
    const evaluationId = getDresEvaluationId();
    const dresBase = getDresBaseUrl().replace(/\/+$/, '');

    if (!sessionId || sessionId === 'None' || sessionId.trim() === '') {
        showTemporaryAlert("⚠️ Bạn chưa có Session ID! Vui lòng đăng nhập hoặc nhập Session ID trong bảng cấu hình.", "warning");
        openDresModal();
        return;
    }

    if (!evaluationId || evaluationId.trim() === '') {
        showTemporaryAlert("⚠️ Chưa có Evaluation ID (ID Cuộc thi)! Vui lòng kiểm tra lại bảng cấu hình.", "warning");
        openDresModal();
        return;
    }

    // 3. Trích xuất thông tin frame đầu tiên trong Export Area
    const firstItem = exportedImages[0];
    console.log("[DRES] Frame to submit from Export Area:", firstItem);

    let rawVideo = "";
    let sec = 0;

    if (firstItem.frameInfo && firstItem.frameInfo.includes('-')) {
        const parts = firstItem.frameInfo.split('-');
        rawVideo = parts[0].trim();
        const parsedSec = parseFloat(parts[1]);
        if (!isNaN(parsedSec)) {
            sec = parsedSec;
        }
    }

    if (!rawVideo && firstItem.videoFramePart) {
        rawVideo = firstItem.videoFramePart.split('/')[0].trim();
    }

    if (sec === 0 && firstItem.frameId !== undefined && firstItem.frameId !== null) {
        const parsedSec = parseFloat(firstItem.frameId);
        if (!isNaN(parsedSec)) {
            sec = parsedSec;
        }
    }

    // Tự động cắt bỏ phần mở rộng như .mp4, .mkv theo đúng quy chế BTC
    const videoId = rawVideo.replace(/\.(mp4|mkv|avi|webm)$/i, '').trim();
    const ms = Math.round(sec * 1000.0);

    if (!videoId) {
        alert("❌ Lỗi: Không thể nhận diện Video ID từ frame đã chọn!");
        return;
    }

    // 4. Xác định Task hiện tại (KIS / Q&A / TRAKE)
    let currentTask = "kis";
    const taskSelect = document.getElementById('global-task-select');
    if (taskSelect && taskSelect.value) {
        currentTask = taskSelect.value.toLowerCase();
    } else if (typeof activeTask !== 'undefined' && activeTask) {
        currentTask = String(activeTask).toLowerCase();
    }

    let body = {};

    if (currentTask === "kis" || currentTask === "kis-v") {
        // KIS-T / KIS-V: Định dạng Video ID + start ms = end ms
        body = {
            "answerSets": [{
                "answers": [{
                    "mediaItemName": videoId,
                    "start": ms,
                    "end": ms
                }]
            }]
        };
    } else if (currentTask === "vqa" || currentTask === "qa" || currentTask === "kis-qa") {
        // Q&A: Định dạng QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>
        let answerText = "";
        const vqaInput = document.querySelector(".vqa-input");
        if (vqaInput && vqaInput.value.trim() !== "") {
            answerText = vqaInput.value.trim();
        } else {
            answerText = prompt(`[Q&A] Nhập câu trả lời (Answer Text) cho video ${videoId} tại ${(ms/1000).toFixed(2)}s:`, "");
            if (answerText === null) {
                showTemporaryAlert("Đã hủy nộp Q&A.", "info");
                return;
            }
            answerText = answerText.trim();
        }

        const qaString = `QA-${answerText}-${videoId}-${ms}`;
        body = {
            "answerSets": [{
                "answers": [{
                    "text": qaString
                }]
            }]
        };
    } else if (currentTask === "trake") {
        // TRAKE: Định dạng TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...
        const matchingFrames = exportedImages
            .filter(item => {
                const v = (item.frameInfo ? item.frameInfo.split('-')[0] : (item.videoFramePart || "")).replace(/\.(mp4|mkv|avi|webm)$/i, '').trim();
                return v === videoId;
            })
            .map(item => {
                const rawId = item.frameId !== undefined ? item.frameId : (item.frameInfo ? item.frameInfo.split('-')[1] : "");
                return String(rawId).trim();
            })
            .filter(Boolean);

        let frameListStr = matchingFrames.join(',');
        if (!frameListStr) {
            frameListStr = prompt(`[TRAKE] Nhập danh sách Semantic Frame IDs cách nhau bởi dấu phẩy cho video ${videoId} (VD: 140,395,820):`, "1");
            if (frameListStr === null) {
                showTemporaryAlert("Đã hủy nộp TRAKE.", "info");
                return;
            }
        }

        const trakeString = `TR-${videoId}-${frameListStr.replace(/\s+/g, '')}`;
        body = {
            "answerSets": [{
                "answers": [{
                    "text": trakeString
                }]
            }]
        };
    }

    const contestURL = `${dresBase}/api/v2/submit/${evaluationId}?session=${sessionId}`;

    console.log("[DRES Submit] Target URL:", contestURL);
    console.log("[DRES Submit] Payload:", JSON.stringify(body, null, 2));

    showTemporaryAlert(`⏳ Đang nộp [${currentTask.toUpperCase()}]: ${videoId} (${(ms/1000).toFixed(2)}s)...`, "info");

    const submitBtn = document.getElementById('submit-button');
    const origBtnHtml = submitBtn ? submitBtn.innerHTML : '';
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Submitting...';
    }

    isSubmitting = true;

    try {
        const response = await fetch(contestURL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            let errorDetail = `HTTP ${response.status}`;
            try {
                const errJson = await response.json();
                errorDetail = errJson.description || errJson.message || errorDetail;
            } catch (e) {
                errorDetail = await response.text();
            }

            if (response.status === 412) {
                showTemporaryAlert("⚠️ Lỗi 412: Nộp trùng với kết quả vừa nộp hoặc thời gian làm bài của task đã hết!", "warning");
                alert("⚠️ THÔNG BÁO TỪ DRES (HTTP 412 - Precondition Failed):\n\nBạn vừa nộp trùng kết quả với lần nộp trước, hoặc thời gian làm bài của câu hỏi đã kết thúc!");
            } else if (response.status === 401) {
                showTemporaryAlert("❌ Lỗi 401: Session ID đã hết hạn hoặc không hợp lệ. Vui lòng đăng nhập lại!", "error");
                alert("❌ LỖI 401: Session ID không hợp lệ hoặc đã hết hạn!\n\nVui lòng đăng nhập lại DRES.");
                openDresModal();
            } else if (response.status === 404) {
                showTemporaryAlert("❌ Lỗi 404: Không tìm thấy Evaluation ID trên DRES!", "error");
                alert(`❌ LỖI 404: Không tìm thấy Evaluation ID (${evaluationId}) trên máy chủ ${dresBase}!`);
                openDresModal();
            } else {
                showTemporaryAlert(`❌ Lỗi nộp bài (${response.status}): ${errorDetail}`, "error");
                alert(`❌ Lỗi từ máy chủ DRES (${response.status}):\n${errorDetail}`);
            }
            return;
        }

        const data = await response.json();
        console.log("[DRES Submit Success]", data);

        const verdict = String(data.submission || data.status || "").toUpperCase();

        if (verdict === 'CORRECT') {
            showTemporaryAlert(`🎉 CHÍNH XÁC (CORRECT)! Video ${videoId} ghi điểm thành công!`, "success");
            alert(`🎉 CHÍNH XÁC (CORRECT)!\n\nVideo: ${videoId}\nThời gian: ${(ms/1000).toFixed(2)}s (${ms} ms)\n\nĐiểm số đã được ghi nhận trên bảng xếp hạng DRES!`);
        } else if (verdict === 'WRONG') {
            showTemporaryAlert(`❌ SAI (WRONG)! Video: ${videoId} (${(ms/1000).toFixed(2)}s). Trừ 10 điểm phạt!`, "error");
            alert(`❌ KẾT QUẢ: SAI (WRONG)!\n\nVideo: ${videoId}\nThời gian: ${(ms/1000).toFixed(2)}s (${ms} ms)\n\n⚠️ Lưu ý: Mỗi lần nộp sai bị trừ 10 điểm phạt theo quy chế cuộc thi. Hãy kiểm tra kỹ lại frame trước khi nộp lại!`);
        } else if (verdict === 'UNDECIDABLE' || verdict === 'INDETERMINATE') {
            showTemporaryAlert(`⏳ ĐANG THẨM ĐỊNH (${verdict})! Ban Giám Khảo đang kiểm tra...`, "info");
            alert(`⏳ KẾT QUẢ: ĐANG THẨM ĐỊNH (${verdict})!\n\nĐáp án đang được Ban Giám Khảo kiểm tra bằng mắt (manual review). Điểm số sẽ được cập nhật sau.`);
        } else {
            showTemporaryAlert(`✅ Đã nộp thành công! Trạng thái: ${verdict || JSON.stringify(data)}`, "success");
        }

    } catch (err) {
        console.error("[DRES Submit Network Error]", err);
        showTemporaryAlert(`❌ Lỗi kết nối DRES: ${err.message}`, "error");
        alert(`❌ Không thể kết nối tới máy chủ DRES:\n${err.message}\n\nEndpoint hiện tại: ${contestURL}\n\nHãy kiểm tra:\n1. Đã kết nối OpenVPN chưa?\n2. IP máy chủ (${dresBase}) có đúng không? Bấm nút 'DRES Login' để đổi IP.`);
    } finally {
        isSubmitting = false;
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = origBtnHtml;
        }
    }
}

window.submit_to_dres_v2 = submit_to_dres_v2;

// ------------------------------------------------------------------------------
// DOM Initialization & Event Binding
// ------------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // 1. Hook DRES Config Button
    const configBtn = document.getElementById('dres-config-btn');
    if (configBtn) {
        configBtn.addEventListener('click', openDresModal);
    }

    // 2. Hook Modal Controls
    const closeBtn = document.getElementById('close-dres-modal');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeDresModal);
    }

    const testBtn = document.getElementById('dres-test-btn');
    if (testBtn) {
        testBtn.addEventListener('click', handleDresTestConnection);
    }

    const saveBtn = document.getElementById('dres-save-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', handleDresSaveConfig);
    }

    const loginBtn = document.getElementById('dres-login-btn');
    if (loginBtn) {
        loginBtn.addEventListener('click', handleDresLogin);
    }

    // 3. Modal background click to close
    const modal = document.getElementById('dresModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeDresModal();
        });
    }

    // 4. Directly hook Submit DRES button to guarantee click works
    const submitBtn = document.getElementById('submit-button');
    if (submitBtn) {
        submitBtn.onclick = (e) => {
            e.preventDefault();
            submit_to_dres_v2();
        };
    }

    // 5. Update initial badge status
    updateDresStatusBadge();

    // 6. Ensure WebSocket handlers still initialize gracefully
    if (typeof connectWebSocket === 'function') try { connectWebSocket(); } catch (e) {}
    if (typeof connectWebSocketcrossing === 'function') try { connectWebSocketcrossing(); } catch (e) {}
    if (typeof connectAlertWebSocket === 'function') try { connectAlertWebSocket(); } catch (e) {}
});
