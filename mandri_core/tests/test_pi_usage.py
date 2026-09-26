import shutil
import subprocess
from importlib.resources import files

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_pi_quota_extension_only_reads_authenticated_provider_quotas():
    script = r"""
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const { default: extension } = await import(pathToFileURL(process.argv[2]).href);
let handler;
extension({ on(event, fn) { assert.equal(event, "session_start"); handler = fn; } });
const urls = [];
const keys = [];
globalThis.fetch = async (url, options) => {
  urls.push(url);
  assert.equal(options.headers.Authorization, "Bearer synthetic-token");
  assert.equal(options.redirect, "error");
  if (url.includes("anthropic")) throw new Error("synthetic provider failure");
  return { ok: true, json: async () => ({ rate_limit: { primary_window: { used_percent: 17 } } }) };
};
const models = ["openai-codex", "openai-codex", "anthropic", "unknown", "api-key"];
let output = "";
  await handler({}, { ui: { notify(value) { output = value; } }, modelRegistry: {
    getAvailable: () => models.map(provider => ({ provider })),
    isUsingOAuth: model => model.provider !== "api-key",
    getApiKeyForProvider: async provider => { keys.push(provider); return "synthetic-token"; },
  } });
assert.equal(urls.length, 2);
assert.deepEqual(keys.sort(), ["anthropic", "openai-codex"]);
const result = JSON.parse(output);
assert.equal(result.type, "mandri_usage");
assert.equal(result.accounts.length, 3);
assert.equal(result.accounts.find(row => row.provider === "anthropic").status, "unavailable");
assert.equal(result.accounts.find(row => row.provider === "unknown").status, "unsupported");
const codex = result.accounts.find(row => row.provider === "openai-codex");
assert.equal(codex.data.rate_limit.primary_window.used_percent, 17);
assert.ok(!output.includes("synthetic-token"));
"""
    resource = str(files("mandri.core").joinpath("resources/pi_usage.ts"))
    subprocess.run(
        ["node", "--input-type=module", "-", resource],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
