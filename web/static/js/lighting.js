(function () {
    "use strict";

    const byId = (id) => document.getElementById(id);
    let bound = false;
    let connected = false;
    let editing = false;
    let requestInFlight = false;
    let backendState = null;

    async function postJSON(url, body) {
        const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        return data;
    }

    function feedback(message, isError) {
        const node = byId("lighting-feedback");
        node.textContent = message || "";
        node.className = `small mt-2 ${isError ? "text-danger" : "text-success"}`;
    }

    function reasonText(reason) {
        if (!reason) return "--";
        const exact = {
            "manual default": "手动默认照度",
            "manual control": "手动控制",
            "manual target": "用户手动设定",
            "automatic fusion control": "自动融合控制",
            "explicit rest intent": "识别到明确休息意图",
            "insufficient information; holding illuminance": "信息不足，保持当前照度",
        };
        if (exact[reason]) return exact[reason];
        if (reason.startsWith("confirmed state: ")) {
            const state = reason.slice(17);
            const states = {tense_active: "紧张烦躁", pleasant_active: "愉悦活跃", low_negative: "低落消沉", calm_relaxed: "平静放松", neutral: "中性过渡"};
            return `已确认状态：${states[state] || state}`;
        }
        if (reason.startsWith("explicit request: ")) {
            const request = reason.slice(18);
            return `明确照明需求：${{off: "关灯", on: "开灯", brighter: "调亮", dimmer: "调暗"}[request] || request}`;
        }
        return reason;
    }

    function setControlAvailability() {
        const disabled = !connected || requestInFlight;
        byId("mode-manual").disabled = disabled;
        byId("mode-automatic").disabled = disabled;
        byId("lux-slider").disabled = disabled || backendState?.mode !== "manual";
    }

    function renderBulb(currentLux) {
        const ratio = Math.max(0, Math.min(1, Number(currentLux || 0) / 600));
        const glass = byId("bulb-glass");
        const halo = byId("bulb-halo");
        if (ratio === 0) {
            glass.style.fill = "rgb(214, 220, 226)";
            glass.style.stroke = "rgb(126, 139, 151)";
            glass.style.filter = "none";
            halo.style.opacity = "0";
            halo.style.transform = "scale(.72)";
            return;
        }
        const eased = Math.pow(ratio, 0.72);
        glass.style.fill = `rgb(${Math.round(174 + 81 * eased)}, ${Math.round(158 + 84 * eased)}, ${Math.round(112 + 76 * eased)})`;
        glass.style.stroke = `rgb(${Math.round(142 + 65 * eased)}, ${Math.round(126 + 64 * eased)}, ${Math.round(86 + 45 * eased)})`;
        glass.style.filter = "url(#bulb-glow)";
        halo.style.opacity = String(0.08 + 0.55 * eased);
        halo.style.transform = `scale(${0.78 + 0.38 * eased})`;
        byId("bulb-blur").setAttribute("stdDeviation", String(3 + 12 * eased));
    }

    function update(lighting, forceEditor = false) {
        if (!lighting) return;
        backendState = {...lighting};
        const manual = lighting.mode === "manual";
        byId("mode-manual").classList.toggle("active", manual);
        byId("mode-automatic").classList.toggle("active", !manual);
        if ((!editing && !requestInFlight) || forceEditor) {
            byId("lux-slider").value = lighting.target_lux ?? 0;
            byId("lighting-target").textContent = lighting.target_lux ?? "--";
        }
        byId("lighting-current").textContent = lighting.current_lux ?? "--";
        byId("lighting-reason").textContent = reasonText(lighting.reason);
        byId("lighting-status").textContent = lighting.status === "confirmed" ? "已确认" : (lighting.status === "holding" ? "保持当前值" : (lighting.status || "--"));
        renderBulb(lighting.current_lux);
        setControlAvailability();
    }

    async function changeMode(mode) {
        if (!connected || requestInFlight) return;
        requestInFlight = true;
        setControlAvailability();
        feedback("正在提交控制模式…", false);
        try {
            const response = await postJSON("/api/lighting/mode", {mode});
            update(response.lighting, true);
            feedback("控制模式已由后端确认", false);
        } catch (error) {
            update(backendState, true);
            feedback(`控制失败：${error.message}`, true);
        } finally {
            requestInFlight = false;
            setControlAvailability();
        }
    }

    async function changeLux(value) {
        if (!connected || requestInFlight || backendState?.mode !== "manual") return;
        editing = false;
        requestInFlight = true;
        setControlAvailability();
        feedback("正在提交仿真照度…", false);
        try {
            const response = await postJSON("/api/lighting/manual", {lux: Number(value)});
            update(response.lighting, true);
            feedback("仿真照度已由后端确认", false);
        } catch (error) {
            update(backendState, true);
            feedback(`调节失败：${error.message}`, true);
        } finally {
            requestInFlight = false;
            setControlAvailability();
        }
    }

    function setConnected(value) {
        connected = Boolean(value);
        if (!connected) {
            editing = false;
            feedback("实时连接不可用，控制已暂时禁用", true);
        }
        setControlAvailability();
    }

    function bind() {
        if (bound) return;
        bound = true;
        byId("mode-manual").addEventListener("click", () => changeMode("manual"));
        byId("mode-automatic").addEventListener("click", () => changeMode("automatic"));
        const slider = byId("lux-slider");
        slider.addEventListener("pointerdown", () => { editing = true; });
        slider.addEventListener("touchstart", () => { editing = true; }, {passive: true});
        slider.addEventListener("keydown", () => { editing = true; });
        slider.addEventListener("input", (event) => {
            editing = true;
            byId("lighting-target").textContent = event.target.value;
        });
        slider.addEventListener("change", (event) => changeLux(event.target.value));
        slider.addEventListener("blur", () => {
            if (editing && !requestInFlight) changeLux(slider.value);
        });
        setControlAvailability();
    }

    window.LightingControls = {bind, update, setConnected, reasonText};
})();
