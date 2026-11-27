(function () {
    "use strict";

    const byId = (id) => document.getElementById(id);
    let bound = false;
    let requestInFlight = false;

    async function postJSON(url, body) {
        const response = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        return data;
    }

    function feedback(message, isError) {
        const node = byId("lighting-feedback");
        node.textContent = message || "";
        node.className = `small mt-2 ${isError ? "text-danger" : "text-success"}`;
    }

    async function changeMode(mode) {
        if (requestInFlight) return;
        requestInFlight = true;
        feedback("正在提交控制模式…", false);
        try {
            const response = await postJSON("/api/lighting/mode", {mode});
            update(response.lighting);
            feedback("控制模式已由后端确认", false);
        } catch (error) {
            feedback(`控制失败：${error.message}`, true);
        } finally {
            requestInFlight = false;
        }
    }

    async function changeLux(value) {
        if (requestInFlight) return;
        requestInFlight = true;
        feedback("正在提交仿真照度…", false);
        try {
            const response = await postJSON("/api/lighting/manual", {lux: Number(value)});
            update(response.lighting);
            feedback("仿真照度已由后端确认", false);
        } catch (error) {
            feedback(`调节失败：${error.message}`, true);
        } finally {
            requestInFlight = false;
        }
    }

    function update(lighting) {
        if (!lighting) return;
        const manual = lighting.mode === "manual";
        byId("mode-manual").classList.toggle("active", manual);
        byId("mode-automatic").classList.toggle("active", !manual);
        byId("lux-slider").disabled = !manual || requestInFlight;
        byId("lux-slider").value = lighting.target_lux ?? 0;
        byId("lighting-target").textContent = lighting.target_lux ?? "--";
        byId("lighting-current").textContent = lighting.current_lux ?? "--";
        byId("lighting-reason").textContent = lighting.reason || "--";
        byId("lighting-status").textContent = lighting.status === "confirmed" ? "已确认" : (lighting.status === "pending" ? "待接入" : (lighting.status || "--"));

        const ratio = Math.max(0, Math.min(1, Number(lighting.current_lux || 0) / 600));
        const warm = Math.round(90 + ratio * 165);
        const glass = byId("bulb-glass");
        const halo = byId("bulb-halo");
        glass.style.fill = `rgb(255, ${warm}, 70)`;
        glass.style.filter = ratio > 0.05 ? "url(#bulb-glow)" : "none";
        halo.style.opacity = String(0.05 + ratio * 0.42);
    }

    function bind() {
        if (bound) return;
        bound = true;
        byId("mode-manual").addEventListener("click", () => changeMode("manual"));
        byId("mode-automatic").addEventListener("click", () => changeMode("automatic"));
        byId("lux-slider").addEventListener("input", (event) => {
            byId("lighting-target").textContent = event.target.value;
        });
        byId("lux-slider").addEventListener("change", (event) => changeLux(event.target.value));
    }

    window.LightingControls = {bind, update};
})();
