if (!_minigpu) var _minigpu = {};
if (!_minigpu.loader) _minigpu.loader = {};

_minigpu.loader.load = function () {
    return new Promise(
        (resolve, reject) => {
            const minigpu_web_js = document.createElement("script");
            minigpu_web_js.src = "assets/packages/minigpu_web/web/minigpu_web.js";
            minigpu_web_js.onerror = reject;
            minigpu_web_js.onload = () => {
                if (runtimeInitialized) resolve();
                Module.onRuntimeInitialized = resolve;
            };
            document.head.append(minigpu_web_js);
        }
    );
}
