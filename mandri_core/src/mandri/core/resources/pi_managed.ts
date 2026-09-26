import registerPermissions from "./pi_permissions.ts";
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

function sessionSnapshot(ctx: any) {
  const manager = ctx.sessionManager;
  const path = manager.getSessionFile();
  const header = manager.getHeader();
  if (!path || !header) return;
  const records = [header, ...manager.getEntries()];
  return { path, content: records.map((entry: any) => JSON.stringify(entry)).join("\n") + "\n" };
}

function replaceFile(path: string, content: string) {
  const temporary = `${path}.${randomUUID()}.tmp`;
  try {
    writeFileSync(temporary, content, { flag: "wx", mode: 0o600 });
    renameSync(temporary, path);
  } finally {
    rmSync(temporary, { force: true });
  }
}

function recordLeaf(_event: any, ctx: any) {
  const manager = ctx.sessionManager;
  const path = manager.getSessionFile();
  const header = manager.getHeader();
  if (!path || !header) return;
  const pointer = `${path}.mandri-leaf`;
  if (!existsSync(path)) {
    rmSync(pointer, { force: true });
    return;
  }
  const stat = statSync(path, { bigint: true });
  replaceFile(pointer, JSON.stringify({
    sessionId: header.id,
    leafId: manager.getLeafId(),
    size: String(stat.size),
    mtimeNs: String(stat.mtimeNs),
  }) + "\n");
}

function checkpoint(_event: any, ctx: any) {
  const path = ctx.sessionManager.getSessionFile();
  if (!path) return;
  const pending = `${path}.mandri-pending`;
  if (existsSync(path)) {
    rmSync(pending, { force: true });
    return;
  }
  const snapshot = sessionSnapshot(ctx);
  if (!snapshot) return;
  mkdirSync(dirname(path), { recursive: true });
  replaceFile(pending, snapshot.content);
}

export default function (pi: any) {
  registerPermissions(pi);
  pi.on("session_start", recordLeaf);
  pi.on("session_tree", (event: any, ctx: any) => {
    recordLeaf(event, ctx);
    ctx.ui.setStatus("_mandri_history_changed", "reset");
  });
  for (const event of ["session_start", "session_info_changed", "model_select", "before_agent_start", "turn_end"]) {
    pi.on(event, checkpoint);
  }
  pi.on("session_shutdown", (_event: any, ctx: any) => {
    const path = ctx.sessionManager.getSessionFile();
    if (path && existsSync(path)) {
      rmSync(`${path}.mandri-pending`, { force: true });
      rmSync(`${path}.mandri-leaf`, { force: true });
      return;
    }
    const snapshot = sessionSnapshot(ctx);
    if (!snapshot) return;
    mkdirSync(dirname(snapshot.path), { recursive: true });
    try {
      writeFileSync(snapshot.path, snapshot.content, { flag: "wx", mode: 0o600 });
    } catch (error: any) {
      if (error.code !== "EEXIST") throw error;
    }
    rmSync(`${snapshot.path}.mandri-pending`, { force: true });
    rmSync(`${snapshot.path}.mandri-leaf`, { force: true });
  });
}
