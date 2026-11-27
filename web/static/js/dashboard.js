(function () {
    "use strict";

    const byId = (id) => document.getElementById(id);
    const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));
    const fmt = (value, digits = 1) => value === null || value === undefined ? "--" : Number(value).toFixed(digits);
    const labels = {
        unknown: "未知", standing: "站立", sitting: "坐姿", lying_candidate: "卧姿候选", bending: "弯腰",
        still: "持续静止", low_activity: "低活动", normal_activity: "正常活动", high_activity: "高活动",
        happy: "开心候选", sad: "悲伤候选", angry: "愤怒候选", surprised: "惊讶候选", neutral: "中性候选",
        unavailable: "不可用", no_speech: "无语音", speaking: "正在说话", processing: "正在转写",
        transcribed: "已转写", untranscribed: "未获得转写", normal_intensity: "正常强度",
        high_intensity: "高强度", low_intensity: "低强度", walking: "行走", pacing_candidate: "踱步候选",
        repetitive_motion: "重复运动", standing_up: "起身", sitting_down: "坐下", manual: "手动", automatic: "自动",
    };
    let chart = null;
    let socket = null;
    let reconnectDelay = 500;
    let lastMessageAt = 0;

    function translated(value) {
        return value === null || value === undefined ? "--" : (labels[value] || String(value));
    }

    function setConnection(connected, text) {
        byId("connection-text").textContent = text;
        const dot = byId("connection-dot");
        dot.className = `status-dot ${connected ? "status-dot-animated bg-green" : "bg-red"}`;
    }

    function setBadge(id, available, stale) {
        const node = byId(id);
        node.textContent = available ? "实时" : (stale ? "过期/不可用" : "不可用");
        node.className = `badge ${available ? "bg-green-lt" : "bg-secondary-lt"}`;
    }

    function setProgress(id, textId, score) {
        const available = score !== null && score !== undefined;
        byId(id).style.width = `${available ? clamp01(score) * 100 : 0}%`;
        byId(textId).textContent = available ? `${(clamp01(score) * 100).toFixed(0)}%` : "--";
    }

    function poseReliability(reliability) {
        if (!reliability) return null;
        return Number.isFinite(Number(reliability.overall)) ? Number(reliability.overall) : null;
    }

    function updatePose(module, performance) {
        setBadge("pose-status", module.available, module.stale);
        const analysis = module.available ? module.analysis : null;
        byId("pose-posture").textContent = translated(analysis?.posture);
        byId("pose-motion").textContent = translated(analysis?.motion_state);
        const behaviors = analysis?.behaviors || {};
        const active = Object.entries(behaviors).filter(([, value]) => value === true).map(([name]) => translated(name));
        byId("pose-behavior").textContent = active.length ? active.join("、") : (analysis ? "未检测到" : "--");
        byId("pose-fps").textContent = fmt(performance.pose_fps);
        byId("pose-ms").textContent = fmt(module.inference_ms);
        setProgress("pose-reliability", "pose-reliability-text", poseReliability(analysis?.reliability));
    }

    function updateFace(module, performance) {
        setBadge("face-status", module.available, module.stale);
        const analysis = module.available ? module.analysis : null;
        byId("face-expression").textContent = translated(analysis?.stable_expression);
        byId("face-activity").textContent = fmt(analysis?.face_activity_score, 2);
        const head = analysis?.head_pose;
        byId("face-head").textContent = head ? `${fmt(head.yaw)}° / ${fmt(head.pitch)}° / ${fmt(head.roll)}°` : "--";
        byId("face-fps").textContent = fmt(performance.face_fps);
        byId("face-ms").textContent = fmt(module.inference_ms);
        setProgress("face-reliability", "face-reliability-text", analysis?.reliability?.score);
    }

    function eventRows(events) {
        if (!Array.isArray(events) || !events.length) {
            return '<span class="text-secondary">当前无超过阈值的白名单事件</span>';
        }
        return events.map((event) => {
            const label = event.display_name || event.name || event.category_name || event.label || "声音事件";
            const score = event.score ?? event.confidence;
            const percent = clamp01(score) * 100;
            return `<div class="sound-event"><div><span>${escapeHtml(label)}</span><strong>${percent.toFixed(0)}%</strong></div><div class="progress progress-sm"><div class="progress-bar bg-orange" style="width:${percent}%"></div></div></div>`;
        }).join("");
    }

    function escapeHtml(value) {
        const node = document.createElement("div");
        node.textContent = value === null || value === undefined ? "" : String(value);
        return node.innerHTML;
    }

    function updateAudio(module) {
        setBadge("audio-status", module.available, module.stale);
        const analysis = module.analysis || null;
        byId("speech-active").textContent = analysis?.speech_active === true ? "检测到" : (analysis?.speech_active === false ? "未检测到" : "--");
        byId("speech-state").textContent = translated(analysis?.speech_state);
        byId("speech-intensity").textContent = translated(analysis?.expression_intensity);
        const reliability = analysis?.reliability?.score;
        byId("audio-reliability").textContent = reliability === null || reliability === undefined ? "--" : `${(clamp01(reliability) * 100).toFixed(0)}%`;
        const buffer = analysis?.buffer_status;
        byId("audio-loss").textContent = buffer ? `${buffer.dropped_blocks ?? 0} / ${buffer.input_overflows ?? 0} / ${buffer.dropped_asr_segments ?? 0}` : "--";

        const transcription = analysis?.transcription;
        byId("transcript-text").textContent = transcription?.text || "暂无有效转写";
        if (transcription) {
            const current = transcription.is_current ? "当前有效" : "最近历史结果";
            const parts = [current, `年龄 ${fmt(transcription.age_seconds)} s`];
            if (transcription.observed_time !== null && transcription.observed_time !== undefined) parts.push(`观测 ${fmt(transcription.observed_time, 3)} s（单调时钟）`);
            if (transcription.language) parts.push(`语言 ${transcription.language}`);
            if (transcription.emotion) parts.push(`模型情感标签 ${transcription.emotion}`);
            if (transcription.event) parts.push(`模型事件标签 ${transcription.event}`);
            byId("transcript-meta").textContent = parts.join(" · ");
        } else {
            byId("transcript-meta").textContent = "历史转写与当前语音状态分开显示";
        }
        const sound = analysis?.sound_event_result;
        byId("sound-events").innerHTML = sound?.is_current ? eventRows(sound.events) : eventRows([]);
    }

    const quadrantPlugin = {
        id: "quadrants",
        beforeDraw(chartInstance) {
            const {ctx, chartArea, scales} = chartInstance;
            if (!chartArea) return;
            const x0 = scales.x.getPixelForValue(0);
            const y0 = scales.y.getPixelForValue(0);
            const {left, right, top, bottom} = chartArea;
            ctx.save();
            [[left, top, x0-left, y0-top, "rgba(70,160,105,.07)"], [x0, top, right-x0, y0-top, "rgba(240,170,65,.07)"],
             [left, y0, x0-left, bottom-y0, "rgba(80,130,210,.07)"], [x0, y0, right-x0, bottom-y0, "rgba(190,95,115,.07)"]]
                .forEach(([x, y, w, h, color]) => { ctx.fillStyle = color; ctx.fillRect(x, y, w, h); });
            ctx.fillStyle = "rgba(70,82,96,.55)"; ctx.font = "12px sans-serif";
            ctx.fillText("紧张烦躁", left + 8, top + 16); ctx.fillText("愉悦活跃", x0 + 8, top + 16);
            ctx.fillText("低落消沉", left + 8, y0 + 18); ctx.fillText("平静放松", x0 + 8, y0 + 18);
            ctx.restore();
        },
    };

    function initChart() {
        if (!window.Chart) return;
        chart = new Chart(byId("russell-chart"), {
            type: "scatter",
            plugins: [quadrantPlugin],
            data: {datasets: [
                {label: "最近轨迹", data: [], showLine: true, borderColor: "rgba(32,107,196,.45)", pointRadius: 2, tension: .2},
                {label: "当前", data: [], backgroundColor: "#206bc4", pointRadius: 6},
            ]},
            options: {responsive: true, maintainAspectRatio: false, animation: false,
                scales: {x: {min: -1, max: 1, title: {display: true, text: "Valence"}}, y: {min: -1, max: 1, title: {display: true, text: "Arousal"}}},
                plugins: {legend: {display: false}},
            },
        });
    }

    function updateFusion(fusion) {
        const available = Boolean(fusion?.available);
        byId("fusion-status").textContent = available ? "可用" : "待接入";
        byId("fusion-status").className = `badge ${available ? "bg-green-lt" : "bg-yellow-lt"}`;
        byId("fusion-valence").textContent = fmt(fusion?.valence, 2);
        byId("fusion-arousal").textContent = fmt(fusion?.arousal, 2);
        byId("fusion-candidate").textContent = translated(fusion?.state_candidate);
        byId("fusion-message").textContent = fusion?.message || "--";
        const base = fusion?.base_weights || {};
        const effectiveV = fusion?.effective_weights?.valence || {};
        const effectiveA = fusion?.effective_weights?.arousal || {};
        byId("fusion-weights").innerHTML = Object.keys(base).map((name) =>
            `<span>${escapeHtml(name)}：基础 ${fmt(base[name], 2)} / V ${fmt(effectiveV[name], 2)} / A ${fmt(effectiveA[name], 2)}</span>`).join("");
        if (chart) {
            const trajectory = Array.isArray(fusion?.trajectory) ? fusion.trajectory.slice(-30) : [];
            chart.data.datasets[0].data = trajectory.filter((p) => Number.isFinite(p.valence) && Number.isFinite(p.arousal)).map((p) => ({x: p.valence, y: p.arousal}));
            chart.data.datasets[1].data = available && Number.isFinite(fusion.valence) && Number.isFinite(fusion.arousal) ? [{x: fusion.valence, y: fusion.arousal}] : [];
            chart.update("none");
        }
    }

    function drawWaveform(waveform) {
        const canvas = byId("waveform-canvas");
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(rect.width * ratio));
        canvas.height = Math.max(1, Math.round(rect.height * ratio));
        const ctx = canvas.getContext("2d");
        ctx.scale(ratio, ratio);
        const width = rect.width, height = rect.height;
        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = "#dce3ea"; ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
        const values = waveform?.available ? waveform.values : [];
        byId("waveform-status").textContent = waveform?.available ? `最近 ${waveform.duration_seconds} 秒压缩波形` : "不可用";
        if (!values?.length) {
            ctx.fillStyle = "#667382"; ctx.font = "14px sans-serif"; ctx.fillText("暂无连续音频数据", 12, height / 2 - 10); return;
        }
        ctx.strokeStyle = "#206bc4"; ctx.lineWidth = 1.2; ctx.beginPath();
        values.forEach((value, index) => {
            const x = values.length === 1 ? 0 : index / (values.length - 1) * width;
            const y = height / 2 - Math.max(-1, Math.min(1, Number(value))) * height * .44;
            if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    function updateHistory(history) {
        const body = byId("history-body");
        if (!Array.isArray(history) || !history.length) {
            body.innerHTML = '<tr><td colspan="3" class="text-center text-secondary">暂无记录</td></tr>'; return;
        }
        body.innerHTML = history.map((item) => `<tr><td>${escapeHtml(item.time || item.timestamp || "--")}</td><td>${escapeHtml(item.event || "--")}</td><td class="text-end">${item.target_lux === null || item.target_lux === undefined ? "--" : `${escapeHtml(item.target_lux)} lux`}</td></tr>`).join("");
    }

    function updateDashboard(data) {
        lastMessageAt = Date.now();
        setConnection(true, data.system?.status === "degraded" ? "服务降级" : "已连接");
        const performance = data.system?.performance || {};
        byId("host-cpu").textContent = `${fmt(performance.host_cpu_percent)}%`;
        byId("host-memory").textContent = `${fmt(performance.host_memory_percent)}%`;
        byId("system-time").textContent = data.timestamp ? new Date(data.timestamp).toLocaleTimeString("zh-CN", {hour12: false}) : "--";
        byId("camera-fps").textContent = fmt(performance.camera_fps);
        byId("process-cpu").textContent = `${fmt(performance.process_cpu_percent)}%`;
        byId("process-memory").textContent = `${fmt(performance.process_memory_mb)} MB`;
        updatePose(data.pose || {}, performance);
        updateFace(data.face || {}, performance);
        updateAudio(data.audio || {});
        updateFusion(data.fusion || {});
        drawWaveform(data.waveform || {});
        window.LightingControls.update(data.lighting);
        byId("summary-state").textContent = translated(data.fusion?.state_candidate);
        byId("summary-hint").textContent = data.fusion?.message || "--";
        byId("summary-lux").textContent = data.lighting?.current_lux ?? "--";
        byId("summary-mode").textContent = translated(data.lighting?.mode);
        byId("summary-light-reason").textContent = data.lighting?.reason || "--";
        updateHistory(data.history);
        const errors = data.system?.errors || {};
        const alert = byId("global-alert");
        const entries = Object.entries(errors);
        alert.classList.toggle("d-none", entries.length === 0);
        alert.textContent = entries.length ? `部分感知模块不可用：${entries.map(([name, message]) => `${name}: ${message}`).join("；")}` : "";
    }

    function connect() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        socket = new WebSocket(`${protocol}//${location.host}/ws/monitor`);
        socket.addEventListener("open", () => { reconnectDelay = 500; setConnection(true, "已连接"); });
        socket.addEventListener("message", (event) => {
            try { updateDashboard(JSON.parse(event.data)); }
            catch (error) { console.error("Invalid monitor payload", error); }
        });
        socket.addEventListener("close", () => {
            setConnection(false, "连接已断开");
            window.setTimeout(connect, reconnectDelay);
            reconnectDelay = Math.min(5000, reconnectDelay * 1.7);
        });
        socket.addEventListener("error", () => socket.close());
    }

    document.addEventListener("DOMContentLoaded", () => {
        window.LightingControls.bind();
        initChart();
        connect();
        window.setInterval(() => {
            if (lastMessageAt && Date.now() - lastMessageAt > 2000) setConnection(false, "数据已过期");
        }, 500);
    });
})();
