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
        pleasant_active: "愉悦活跃候选", tense_active: "紧张烦躁候选", low_negative: "低落消沉候选", calm_relaxed: "平静放松候选",
    };
    let chart = null;
    let socket = null;
    let reconnectDelay = 500;
    let lastMessageAt = 0;
    let realtimeInvalidated = true;

    function translated(value) {
        return value === null || value === undefined ? "--" : (labels[value] || String(value));
    }

    function fusionMessage(value) {
        return {"fusion active": "有效观测已融合", "waiting for new observation": "等待新观测"}[value] || value || "--";
    }

    function setConnection(connected, text, enableControls = connected) {
        byId("connection-text").textContent = text;
        const dot = byId("connection-dot");
        dot.className = `status-dot ${connected ? "status-dot-animated bg-green" : "bg-red"}`;
        window.LightingControls.setConnected(enableControls);
    }

    function setBadge(id, module) {
        const node = byId(id);
        let text = "实时";
        let style = "bg-green-lt";
        if (!module.device_available) { text = "设备不可用"; style = "bg-secondary-lt"; }
        else if (module.analysis?.audio_available === false) { text = "数据不可用"; style = "bg-secondary-lt"; }
        else if (module.stale) { text = "数据已过期"; style = "bg-yellow-lt"; }
        else if (!module.available) { text = "数据不可用"; style = "bg-secondary-lt"; }
        else if (module.target_detected === false) { text = "未检测到目标"; style = "bg-blue-lt"; }
        else if (module.observation_reliable === false) { text = "信息不足"; style = "bg-yellow-lt"; }
        node.textContent = text;
        node.className = `badge ${style}`;
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

    function updatePose(module, performance, source) {
        const staticImage = source?.type === "image";
        setBadge("pose-status", module);
        if (staticImage && module.available) {
            byId("pose-status").textContent = "静态分析完成";
            byId("pose-status").className = "badge bg-blue-lt";
        }
        const analysis = module.available ? module.analysis : null;
        const personAbsent = module.available && module.target_detected === false;
        byId("pose-posture").textContent = personAbsent ? "未检测到人体" : translated(analysis?.posture);
        byId("pose-motion").textContent = staticImage ? "不适用" : (personAbsent ? "--" : translated(analysis?.motion_state));
        const behaviors = analysis?.behaviors || {};
        const active = Object.entries(behaviors).filter(([, value]) => value === true).map(([name]) => translated(name));
        byId("pose-behavior").textContent = staticImage ? "不适用" : (active.length ? active.join("、") : (analysis ? "未检测到" : "--"));
        byId("pose-performance").textContent = staticImage && module.inference_ms !== null && module.inference_ms !== undefined
            ? `${fmt(module.inference_ms)} ms（单次）`
            : `${fmt(performance.pose_fps)} FPS / ${fmt(module.inference_ms)} ms`;
        setProgress("pose-reliability", "pose-reliability-text", poseReliability(analysis?.reliability));
        setProgress("pose-activity", "pose-activity-text", staticImage ? null : analysis?.activity_score);
        if (staticImage) byId("pose-activity-text").textContent = "不适用";
    }

    function updateFace(module, performance, source) {
        const staticImage = source?.type === "image";
        setBadge("face-status", module);
        if (staticImage && module.available) {
            byId("face-status").textContent = "静态分析完成";
            byId("face-status").className = "badge bg-blue-lt";
        }
        const analysis = module.available ? module.analysis : null;
        const faceAbsent = module.available && module.target_detected === false;
        byId("face-expression-label").textContent = staticImage ? "瞬时表情候选" : "稳定表情候选";
        byId("face-expression").textContent = faceAbsent ? "未检测到人脸" : translated(staticImage ? analysis?.raw_expression : analysis?.stable_expression);
        byId("face-activity").textContent = staticImage ? "不适用" : fmt(analysis?.face_activity_score, 2);
        const head = analysis?.head_pose;
        byId("face-head").textContent = head ? `${fmt(head.yaw)}° / ${fmt(head.pitch)}° / ${fmt(head.roll)}°` : "--";
        byId("face-performance").textContent = staticImage && module.inference_ms !== null && module.inference_ms !== undefined
            ? `${fmt(module.inference_ms)} ms（单次）`
            : `${fmt(performance.face_fps)} FPS / ${fmt(module.inference_ms)} ms`;
        const input = analysis?.detection_input;
        byId("face-input-region").textContent = input?.source === "pose_head_roi"
            ? `Pose头部ROI ${(clamp01(input.reliability) * 100).toFixed(0)}%`
            : (analysis ? "全图回退" : "--");
        setProgress("face-reliability", "face-reliability-text", analysis?.reliability?.score);
        ["happy", "sad", "angry", "surprised", "neutral"].forEach((name) => {
            setProgress(`face-score-${name}`, `face-score-${name}-text`, analysis?.expression_scores?.[name]);
        });
        const debug = analysis?.expression_debug;
        byId("face-expression-debug").textContent = debug
            ? `规则竞争：${debug.sadness_pattern === "crying_like" ? "哭泣样组合" : (debug.sadness_pattern === "mild_sad" ? "轻度悲伤组合" : "无悲伤组合")}；哭泣样分数 ${fmt(debug.cry_score, 2)}；开心抑制 ${(clamp01(debug.happy_suppression) * 100).toFixed(0)}%`
            : "规则竞争调试：--";
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

    function updateAudio(module, semantic, source) {
        const staticImage = source?.type === "image";
        if (staticImage) {
            byId("audio-status").textContent = "图片模式不适用";
            byId("audio-status").className = "badge bg-secondary-lt";
            byId("speech-active").textContent = "不适用";
            byId("speech-state").textContent = "不适用";
            byId("speech-intensity").textContent = "不适用";
            byId("audio-reliability").textContent = "不适用";
            byId("audio-loss").textContent = "不适用";
            byId("transcript-text").textContent = "静态图片不包含音频时序分析";
            byId("transcript-meta").textContent = "不适用";
            byId("sound-events").innerHTML = '<span class="text-secondary">图片模式不适用</span>';
            byId("semantic-result").textContent = "图片模式不适用";
            byId("semantic-alert").classList.add("d-none");
            return;
        }
        setBadge("audio-status", module);
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
            if (Number.isFinite(Number(transcription.age_seconds))) {
                const observed = new Date(Date.now() - Number(transcription.age_seconds) * 1000);
                parts.push(`最近转写 ${observed.toLocaleTimeString("zh-CN", {hour12: false})}`);
            }
            if (transcription.language) parts.push(`语言 ${transcription.language}`);
            if (transcription.emotion) parts.push(`模型情感标签 ${transcription.emotion}`);
            if (transcription.event) parts.push(`模型事件标签 ${transcription.event}`);
            byId("transcript-meta").textContent = parts.join(" · ");
        } else {
            byId("transcript-meta").textContent = "历史转写与当前语音状态分开显示";
        }
        const sound = analysis?.sound_event_result;
        if (!module.available) {
            byId("sound-events").innerHTML = '<span class="text-secondary">声音事件识别不可用</span>';
        } else if (!sound) {
            byId("sound-events").innerHTML = '<span class="text-secondary">等待有效声音分类窗口</span>';
        } else if (!sound.is_current) {
            byId("sound-events").innerHTML = '<span class="text-secondary">声音事件结果已过期</span>';
        } else {
            byId("sound-events").innerHTML = eventRows(sound.events);
        }
        const categoryNames = {
            negative_emotion: "消极情绪表达", depressive_expression: "低落相关表达",
            agitation: "烦躁或激越表达", self_harm_suicide: "自伤或自杀相关表达",
            threat_conflict: "冲突或威胁表达", positive_emotion: "正向情绪表达",
            rest_intent: "休息意图", lighting_request: "照明需求",
        };
        const categories = semantic?.categories || [];
        byId("semantic-result").textContent = categories.length
            ? categories.map((item) => `${categoryNames[item.category] || item.category}：${(item.matched_phrases || []).join("、")}`).join("；")
            : "当前未匹配语义规则";
        const alerts = semantic?.alerts || [];
        const alertNode = byId("semantic-alert");
        alertNode.classList.toggle("d-none", alerts.length === 0);
        alertNode.textContent = alerts.length
            ? `人工核查提示：${alerts.map((item) => `${categoryNames[item.category] || item.category}（匹配：${item.matched_phrases.join("、")}）`).join("；")}。该提示不代表诊断，也不直接参与自动灯光调节。`
            : "";
    }

    const quadrantPlugin = {
        id: "quadrants",
        beforeDatasetsDraw(chartInstance) {
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
            const radius = Math.min(
                Math.abs(scales.x.getPixelForValue(1) - x0),
                Math.abs(scales.y.getPixelForValue(1) - y0)
            );
            ctx.strokeStyle = "rgba(70,82,96,.42)"; ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.arc(x0, y0, radius, 0, Math.PI * 2); ctx.stroke();
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
        const statusText = {confirmed: "已确认", pending: "待确认", insufficient: "信息不足", snapshot: "静态快照"}[fusion?.status];
        byId("fusion-status").textContent = statusText || (available ? "可用" : "信息不足");
        byId("fusion-status").className = `badge ${available && fusion?.status === "confirmed" ? "bg-green-lt" : "bg-yellow-lt"}`;
        byId("fusion-valence").textContent = fmt(fusion?.valence, 2);
        byId("fusion-arousal").textContent = fmt(fusion?.arousal, 2);
        byId("fusion-candidate").textContent = translated(fusion?.state_candidate);
        byId("fusion-message").textContent = fusionMessage(fusion?.message);
        const base = fusion?.base_weights || {};
        const effectiveV = fusion?.effective_weights?.valence || {};
        const effectiveA = fusion?.effective_weights?.arousal || {};
        const modalityNames = {pose: "人体动作", face: "面部表情", speech_content: "语音内容", speech_prosody: "语音韵律"};
        const weightCell = (label, value, color) => {
            const percent = value === null || value === undefined ? null : clamp01(value) * 100;
            return `<div class="weight-cell"><div class="weight-caption"><span>${label}</span><span>${percent === null ? "--" : `${percent.toFixed(0)}%`}</span></div><div class="progress"><div class="progress-bar ${color}" style="width:${percent ?? 0}%"></div></div></div>`;
        };
        byId("fusion-weights").innerHTML = Object.keys(base).map((name) =>
            `<div class="weight-row"><strong>${escapeHtml(modalityNames[name] || name)}</strong><div class="weight-bars">${weightCell("基础", base[name], "bg-secondary")}${weightCell("V", effectiveV[name], "bg-blue")}${weightCell("A", effectiveA[name], "bg-orange")}</div></div>`).join("");
        if (chart) {
            const allPoints = Array.isArray(fusion?.trajectory) ? fusion.trajectory : [];
            const timedPoints = allPoints.filter((point) => Number.isFinite(Number(point.timestamp)));
            const latestTime = timedPoints.length ? Math.max(...timedPoints.map((point) => Number(point.timestamp))) : null;
            const trajectory = latestTime === null ? [] : timedPoints.filter((point) => Number(point.timestamp) >= latestTime - 30);
            chart.data.datasets[0].data = trajectory.filter((p) => Number.isFinite(p.valence) && Number.isFinite(p.arousal)).map((p) => ({x: p.valence, y: p.arousal}));
            chart.data.datasets[1].data = available && Number.isFinite(fusion.valence) && Number.isFinite(fusion.arousal) ? [{x: fusion.valence, y: fusion.arousal}] : [];
            chart.update("none");
        }
    }

    function drawWaveform(waveform, source) {
        const canvas = byId("waveform-canvas");
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(rect.width * ratio));
        canvas.height = Math.max(1, Math.round(rect.height * ratio));
        const ctx = canvas.getContext("2d");
        ctx.scale(ratio, ratio);
        const width = rect.width, height = rect.height;
        ctx.clearRect(0, 0, width, height);
        const envelope = waveform?.available ? waveform.envelope : [];
        const duration = Number(waveform?.duration_seconds || 0);
        if (source?.type === "image") {
            byId("waveform-status").textContent = "图片模式不适用";
            ctx.fillStyle = "#667382";
            ctx.font = "14px sans-serif";
            ctx.fillText("静态图片无音频波形", 12, height / 2);
            return;
        }
        byId("waveform-status").textContent = waveform?.available ? `有效窗口 ${duration.toFixed(2)} 秒` : "不可用";
        const left = 36, right = 8, top = 8, bottom = 18;
        const plotWidth = Math.max(1, width - left - right), plotHeight = Math.max(1, height - top - bottom);
        ctx.strokeStyle = "rgba(210,225,238,.24)"; ctx.fillStyle = "#a9bac8"; ctx.font = "10px sans-serif";
        [-1, 0, 1].forEach((value) => {
            const y = top + (1 - (value + 1) / 2) * plotHeight;
            ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
            ctx.fillText(String(value), 8, y + 3);
        });
        ctx.fillText(`-${duration.toFixed(1)}s`, left, height - 4); ctx.fillText("0s", width - 23, height - 4);
        if (!envelope?.length) {
            ctx.fillStyle = "#667382"; ctx.font = "14px sans-serif"; ctx.fillText("暂无连续音频数据", 12, height / 2 - 10); return;
        }
        ctx.strokeStyle = "#58c8ff"; ctx.lineWidth = 1;
        envelope.forEach((pair, index) => {
            const x = left + (index + .5) / envelope.length * plotWidth;
            const minimum = Math.max(-1, Math.min(1, Number(pair[0])));
            const maximum = Math.max(-1, Math.min(1, Number(pair[1])));
            const yMin = top + (1 - (minimum + 1) / 2) * plotHeight;
            const yMax = top + (1 - (maximum + 1) / 2) * plotHeight;
            ctx.beginPath(); ctx.moveTo(x, yMin); ctx.lineTo(x, yMax); ctx.stroke();
        });
    }

    function invalidateRealtime(reason) {
        const alreadyInvalid = realtimeInvalidated;
        realtimeInvalidated = true;
        if (!alreadyInvalid) {
            const unavailable = {available: false, device_available: false, stale: true, analysis: null};
            updatePose(unavailable, {pose_fps: 0});
            updateFace(unavailable, {face_fps: 0});
            updateAudio(unavailable);
            drawWaveform({available: false, duration_seconds: 0, envelope: []});
            byId("camera-fps").textContent = "--";
        }
        window.LightingControls.setConnected(false);
        setConnection(false, reason, false);
    }

    function updateHistory(history) {
        const body = byId("history-body");
        if (!Array.isArray(history) || !history.length) {
            body.innerHTML = '<tr><td colspan="3" class="text-center text-secondary">暂无记录</td></tr>'; return;
        }
        const eventText = (value) => {
            if (!value) return "--";
            if (value.startsWith("confirmed state: ")) return `已确认综合状态：${translated(value.slice(17))}`;
            if (value.startsWith("source switched: ")) return `数据源已切换：${value.slice(17)}`;
            if (value.startsWith("lighting mode: ")) return `照明模式：${translated(value.slice(15))}`;
            if (value.startsWith("semantic alert: ")) {
                const category = value.slice(16);
                return {self_harm_suicide: "敏感语义需人工核查：自伤或自杀相关表达", threat_conflict: "敏感语义需人工核查：冲突或威胁表达"}[category] || "敏感语义需人工核查";
            }
            if (value === "manual illuminance adjusted") return "手动仿真照度已调整";
            return window.LightingControls.reasonText(value);
        };
        body.innerHTML = history.map((item) => `<tr><td>${escapeHtml(item.time || item.timestamp || "--")}</td><td>${escapeHtml(eventText(item.event))}</td><td class="text-end">${item.target_lux === null || item.target_lux === undefined ? "--" : `${escapeHtml(item.target_lux)} lux`}</td></tr>`).join("");
    }

    async function postSource(url, body) {
        const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        return data;
    }

    async function uploadSource(file, kind) {
        if (!file) return;
        byId("source-feedback").textContent = `正在载入并初始化分析：${file.name}`;
        byId("source-feedback").className = "source-feedback px-3 pb-2 text-primary";
        const url = `/api/source/upload?kind=${encodeURIComponent(kind)}&filename=${encodeURIComponent(file.name)}`;
        const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: file});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        updateSource(data.source);
        byId("source-feedback").textContent = kind === "image" ? "图片已载入：仅分析当前静态画面，不累积时序状态。" : "视频已载入并开始分析，可使用暂停、重新播放和停止控制。";
        byId("source-feedback").className = "source-feedback px-3 pb-2 text-success";
    }

    function updateSource(source) {
        if (!source) return;
        const position = Number(source.position_seconds || 0), duration = Number(source.duration_seconds || 0);
        const progress = Number(source.progress);
        byId("source-progress-bar").style.width = `${Number.isFinite(progress) ? Math.max(0, Math.min(100, progress * 100)) : 0}%`;
        const audioNote = source.audio_status === "ffmpeg_unavailable" ? " / FFmpeg 不可用" : (source.audio_status === "no_audio_track" ? " / 无音轨" : (source.audio_status === "extraction_failed" ? " / 音轨提取失败" : ""));
        const stateLabel = {playing: "播放中", paused: "已暂停", stopped: "已停止", finished: "播放结束", ready: "等待分析", analyzed: "静态分析完成"}[source.state] || source.state;
        byId("source-status").textContent = `${source.name || source.type} / ${stateLabel}${duration ? ` / ${position.toFixed(1)}s / ${duration.toFixed(1)}s` : ""}${audioNote}`;
        ["play", "pause", "replay", "stop"].forEach((action) => { byId(`source-${action}`).disabled = source.type !== "video"; });
    }

    function bindSourceControls() {
        const failed = (error) => {
            byId("source-feedback").textContent = `素材操作失败：${error.message}`;
            byId("source-feedback").className = "source-feedback px-3 pb-2 text-danger";
        };
        byId("source-live").addEventListener("click", () => postSource("/api/source/live", {type: "live"}).then((data) => {
            updateSource(data.source);
            byId("source-feedback").textContent = "已切换到实时摄像头和麦克风。";
            byId("source-feedback").className = "source-feedback px-3 pb-2 text-success";
        }).catch(failed));
        byId("source-video-file").addEventListener("change", (event) => uploadSource(event.target.files[0], "video").catch(failed).finally(() => { event.target.value = ""; }));
        byId("source-image-file").addEventListener("change", (event) => uploadSource(event.target.files[0], "image").catch(failed).finally(() => { event.target.value = ""; }));
        const actionText = {play: "视频已开始播放并继续分析。", pause: "视频已暂停；暂停期间不累积时序确认。", replay: "视频已重新播放，旧时序缓存已清除。", stop: "视频播放与分析已停止。"};
        ["play", "pause", "replay", "stop"].forEach((action) => byId(`source-${action}`).addEventListener("click", () => postSource("/api/source/control", {action}).then((data) => {
            updateSource(data.source);
            byId("source-feedback").textContent = actionText[action];
            byId("source-feedback").className = "source-feedback px-3 pb-2 text-secondary";
        }).catch(failed)));
    }

    function updateDashboard(data) {
        lastMessageAt = Date.now();
        realtimeInvalidated = false;
        setConnection(true, data.system?.status === "degraded" ? "服务降级" : "已连接");
        const performance = data.system?.performance || {};
        const source = data.source || {};
        byId("host-cpu").textContent = `${fmt(performance.host_cpu_percent)}%`;
        byId("host-memory").textContent = `${fmt(performance.host_memory_percent)}%`;
        byId("system-time").textContent = data.timestamp ? new Date(data.timestamp).toLocaleTimeString("zh-CN", {hour12: false}) : "--";
        byId("camera-performance-label").textContent = source.type === "image" ? "静态素材分析" : "共享摄像头性能";
        byId("camera-fps").textContent = source.type === "image" ? "单次" : fmt(performance.camera_fps);
        byId("camera-fps-unit").textContent = source.type === "image" ? "推理" : "FPS";
        byId("process-cpu").textContent = `${fmt(performance.process_cpu_percent)}%`;
        byId("process-memory").textContent = `${fmt(performance.process_memory_mb)} MB`;
        updatePose(data.pose || {}, performance, source);
        updateFace(data.face || {}, performance, source);
        updateAudio(data.audio || {}, data.fusion?.semantic || {}, source);
        updateFusion(data.fusion || {});
        drawWaveform(data.waveform || {}, source);
        window.LightingControls.update(data.lighting);
        byId("summary-state").textContent = translated(data.fusion?.confirmed_state || data.fusion?.state_candidate);
        byId("summary-hint").textContent = fusionMessage(data.fusion?.message);
        byId("summary-lux").textContent = data.lighting?.current_lux ?? "--";
        byId("summary-mode").textContent = translated(data.lighting?.mode);
        byId("summary-light-reason").textContent = window.LightingControls.reasonText(data.lighting?.reason);
        updateHistory(data.history);
        updateSource(data.source);
        const errors = data.system?.errors || {};
        const alert = byId("global-alert");
        const entries = Object.entries(errors);
        alert.classList.toggle("d-none", entries.length === 0);
        alert.textContent = entries.length ? `部分感知模块不可用：${entries.map(([name, message]) => `${name}: ${message}`).join("；")}` : "";
    }

    function connect() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        socket = new WebSocket(`${protocol}//${location.host}/ws/monitor`);
        socket.addEventListener("open", () => { reconnectDelay = 500; setConnection(true, "已连接，等待数据", false); });
        socket.addEventListener("message", (event) => {
            try { updateDashboard(JSON.parse(event.data)); }
            catch (error) { console.error("Invalid monitor payload", error); }
        });
        socket.addEventListener("close", () => {
            invalidateRealtime("连接已断开");
            window.setTimeout(connect, reconnectDelay);
            reconnectDelay = Math.min(5000, reconnectDelay * 1.7);
        });
        socket.addEventListener("error", () => socket.close());
    }

    document.addEventListener("DOMContentLoaded", () => {
        window.LightingControls.bind();
        bindSourceControls();
        initChart();
        connect();
        window.setInterval(() => {
            if (lastMessageAt && Date.now() - lastMessageAt > 2000) invalidateRealtime("数据已过期");
        }, 500);
    });
})();
