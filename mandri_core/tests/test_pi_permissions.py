import shutil
import subprocess
from importlib.resources import files

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_managed_permissions_gate_tools_and_acknowledge_changes():
    script = r"""
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const { default: extension } = await import(pathToFileURL(process.argv[2]).href);
const handlers = {};
let change;
const statuses = [];
const requests = [];
extension({
  on(name, fn) { handlers[name] = fn; },
  registerCommand(name, command) {
    assert.equal(name, "mandri-permissions"); change = command.handler;
  },
});
const ctx = { hasUI: true, ui: {
  setStatus(key, value) { statuses.push([key, value]); },
  async confirm(title, content) { requests.push([title, content]); return false; },
} };
const tool = name => handlers.tool_call({ toolName: name, input: { path: "synthetic.txt" } }, ctx);
assert.equal(await tool("read"), undefined);
assert.equal((await tool("bash")).block, true);
await change("acceptEdits token1", ctx);
assert.equal(await tool("write"), undefined);
assert.equal((await tool("bash")).block, true);
await change("plan token2", ctx);
assert.equal((await tool("write")).block, true);
assert.equal((await tool("custom_tool")).block, true);
assert.equal(await tool("grep"), undefined);
await change("bypassPermissions token3", ctx);
assert.equal(await tool("bash"), undefined);
await change("default token4", ctx);
assert.equal((await tool("write")).block, true);
assert.deepEqual(statuses.at(-1), ["_mandri_permissions", "token4:default"]);
assert.equal(requests.length, 3);
await assert.rejects(change("unknown token5", ctx));
assert.equal(statuses.length, 4);
"""
    subprocess.run(
        [
            "node",
            "--input-type=module",
            "-",
            str(files("mandri.core").joinpath("resources/pi_permissions.ts")),
        ],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
