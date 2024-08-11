# modify_js.cmake.in
file(READ "minigpu_web.js" JS_CONTENT)
string(REPLACE 
    "audioWorklet.addModule(\"minigpu_web.aw.js\")" 
    "audioWorklet.addModule(locateFile(\"minigpu_web.aw.js\"))" 
    MODIFIED_JS_CONTENT "${JS_CONTENT}"
)
file(WRITE "minigpu_web.js" "${MODIFIED_JS_CONTENT}")
