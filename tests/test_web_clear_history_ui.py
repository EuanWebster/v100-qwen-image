"""The History tab's delete confirmation, statically AND executed in node.

Static checks pin the markup contract (a visibly named button, the exact
"Also delete the image files from disk" tick box, the status line) and the
no-``alert()``/no-``confirm()`` rule. The node harness then extracts the REAL
deletion functions from app.js verbatim and drives them over fake caches and
fake server answers, asserting what the page shows, what it POSTs and what it
keeps when the server refuses. (Same executed-harness convention as
tests/test_web_static.py uses for the Settings parsers.)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "qwen_engine" / "web_static"


def _read(name: str) -> str:
    path = WEB_DIR / name
    assert path.is_file(), f"missing packaged static asset: {path}"
    return path.read_text(encoding="utf-8")


def _js_function(js: str, signature: str) -> str:
    m = re.search(rf"function {re.escape(signature)}\([\s\S]*?\n  \}}", js)
    assert m, f"{signature} not found in app.js"
    return m.group(0)


def _js_var(js: str, name: str) -> str:
    m = re.search(rf"^  var {re.escape(name)} = [\s\S]*?;$", js, re.M)
    assert m, f"{name} not found in app.js"
    return m.group(0)


# ----------------------------------------------------------- static contract


def test_the_delete_all_button_is_named_disabled_until_finished_history_exists():
    html = _read("index.html")
    button = re.search(r'<button[^>]*id="history-delete-all"[^>]*>', html).group(0)
    assert "btn-danger" in button and "disabled" in button, button
    assert "Delete all history" in html
    assert 'id="history-status"' in html
    status = re.search(r'<p[^>]*id="history-status"[^>]*>', html).group(0)
    assert 'role="status"' in status and 'aria-live="polite"' in status and "hidden" in status


def test_the_confirmation_is_in_the_page_with_the_named_tick_box_unticked():
    html = _read("index.html")
    panel = re.search(r'<div[^>]*id="history-confirm"[^>]*>[\s\S]*?</div>\s*<p class="history-status"',
                      html).group(0)
    assert 'role="group"' in panel and "hidden" in panel
    assert 'id="history-confirm-question"' in panel and 'aria-live="polite"' in panel
    assert '<label for="history-confirm-files">Also delete the image files from disk</label>' in panel
    box = re.search(r'<input[^>]*id="history-confirm-files"[^>]*>', html).group(0)
    assert 'type="checkbox"' in box and "checked" not in box, "unticked by default"
    assert 'id="history-confirm-cancel"' in panel and 'id="history-confirm-yes"' in panel
    css = _read("app.css")
    for selector in (".history-confirm", ".history-confirm-files", ".btn-danger"):
        assert selector in css, selector


def test_no_browser_dialog_ever_and_per_entry_forget_opens_the_confirmation():
    js = _read("app.js")
    for call in ("alert(", "confirm("):
        for line in js.splitlines():
            if call in line:
                stripped = line.strip()
                assert (stripped.startswith("//") or stripped.startswith("*")
                        or "never" in stripped), f"live browser dialog call: {line}"
    history = _js_function(js, "renderHistory")
    assert "openHistoryConfirm(id)" in history
    assert "jobAction(id, 'forget')" not in history
    assert "updateHistoryControls();" in history
    controls = _js_function(js, "updateHistoryControls")
    assert "finished === 0" in controls, "no finished history must disable the button"
    deletion = _js_function(js, "confirmHistoryDeletion")
    assert "API_JOBS_FORGET_ALL" in deletion
    assert "'X-Qwen-UI': '1'" in deletion, "the delete is a write like every other POST"
    assert "JSON.stringify({delete_files: deleteFiles})" in deletion
    assert "parseHistoryDeletion" in deletion, "server counts are strictly parsed"


def test_job_documents_carry_the_image_size_the_confirmation_names():
    js = _read("app.js")
    parse = _js_function(js, "parseJob")
    assert "image_bytes" in parse
    assert f"var API_JOBS_FORGET_ALL = API_JOBS + '/forget-all';" in js
    assert "'qwen_engine.web_jobs.v1'" in _js_function(js, "parseHistoryDeletion")
    # ONE page-wide byte formatter (the model-setup one, binary units): a
    # second definition would silently win by declaration order, which is
    # exactly the merge collision this pins shut.
    assert js.count("function formatBytes(") == 1


HARNESS_VARS = ("API_JOBS", "API_JOBS_FORGET_ALL", "ERROR_SCHEMA", "jobCache",
                "jobSeq", "historyConfirmTarget", "clearingHistory")
HARNESS_FNS = ("jobTerminal", "jobUrl", "setText", "formatBytes",
               "terminalHistoryJobs", "historyImageInfo", "setHistoryStatus",
               "updateHistoryControls", "updateHistoryConfirmButton",
               "openHistoryConfirm", "closeHistoryConfirm",
               "parseHistoryDeletion", "confirmHistoryDeletion")

_HARNESS = """// Built by tests/test_web_clear_history_ui.py: the real app.js deletion
// functions, extracted verbatim, driven over fake job caches and fake server
// answers with the smallest DOM they touch. Nothing here reimplements a
// product rule.
var fs = require("node:fs");
__PARTS__

var NODES = new Map();
function el(id) {
  return {id: id, textContent: "", title: "", hidden: true, disabled: true,
          checked: false, focus: function () { focused = id; },
          addEventListener: function () {}};
}
var focused = null;
function $(id) {
  if (!NODES.has(id)) NODES.set(id, el(id));
  return NODES.get(id);
}
$("history-confirm-files").parentNode = {hidden: false};

var calls = [];
var refreshCalls = 0;
var nextAnswer = {res: {ok: true}, json: {}};
function apiRequest(url, opts) {
  calls.push({url: url, method: opts.method,
              contentType: opts.headers['Content-Type'], ui: opts.headers['X-Qwen-UI'],
              body: opts.body});
  return Promise.resolve(nextAnswer);
}
function refreshJobs() { refreshCalls++; return Promise.resolve(true); }

function reset() {
  Object.keys(jobCache).forEach(function (id) { delete jobCache[id]; });
  calls = []; refreshCalls = 0; focused = null;
  historyConfirmTarget = null; clearingHistory = false;
  NODES.forEach(function (node, id) { NODES.set(id, el(id)); });
  $("history-confirm-files").parentNode = {hidden: false};
}
function job(id, state, imageBytes) {
  return {job_id: id, state: state,
          image_url: state === "succeeded" ? jobUrl(id) + "/image" : null,
          image_bytes: state === "succeeded" ? imageBytes : null};
}
function snapshot() {
  return {confirmHidden: $("history-confirm").hidden,
          question: $("history-confirm-question").textContent,
          filesHidden: $("history-confirm-files").parentNode.hidden,
          filesChecked: $("history-confirm-files").checked,
          yesText: $("history-confirm-yes").textContent,
          deleteAllDisabled: $("history-delete-all").disabled,
          deleteAllTitle: $("history-delete-all").title,
          status: $("history-status").textContent,
          statusHidden: $("history-status").hidden,
          focused: focused,
          calls: calls, refreshCalls: refreshCalls,
          cached: Object.keys(jobCache).sort()};
}
function pump() { return new Promise(function (r) { setTimeout(r, 30); }); }

var A = "a".repeat(32), B = "b".repeat(32), C = "c".repeat(32);
var out = {};

out.emptyDisabled = (function () {
  reset();
  updateHistoryControls();
  return {deleteAllDisabled: $("history-delete-all").disabled};
})();

out.allUnticked = (function () {
  reset();
  jobCache[A] = job(A, "succeeded", 1572864 + 1048576);   // exactly 2.5 MiB
  jobCache[B] = job(B, "failed", null);
  jobCache[C] = job(C, "queued", null);
  updateHistoryControls();
  var beforeOpen = {deleteAllDisabled: $("history-delete-all").disabled};
  openHistoryConfirm("all");
  return Object.assign(beforeOpen, snapshot());
})();

out.tickedChangesTheButton = (function () {
  reset();
  jobCache[A] = job(A, "succeeded", 1048576 * 3);
  jobCache[B] = job(B, "cancelled", null);
  openHistoryConfirm("all");
  $("history-confirm-files").checked = true;
  updateHistoryConfirmButton();
  return snapshot();
})();

var A = "a".repeat(32), B = "b".repeat(32), C = "c".repeat(32);

// The confirm path is promise-driven and clears a shared in-flight flag, so
// the async scenarios run one after another (never interleaved): each resets
// global state, drives the click, waits for the request to settle, snapshots.
function step(setup) {
  return function () {
    reset();
    setup();
    return pump();
  };
}
var sequence = Promise.resolve();
function run(name, setup) {
  sequence = sequence.then(function () {
    return step(setup)().then(function () { out[name] = snapshot(); });
  });
}

run("confirmFilesDeletes", function () {
  jobCache[A] = job(A, "succeeded", 1048576 * 3);
  jobCache[B] = job(B, "cancelled", null);
  jobCache[C] = job(C, "running", null);
  openHistoryConfirm("all");
  $("history-confirm-files").checked = true;
  nextAnswer = {res: {ok: true}, json: {schema: "qwen_engine.web_jobs.v1",
                                        forgotten: 2, deleted_files: 3, skipped_live: 1}};
  confirmHistoryDeletion();
});

run("confirmKeepsFiles", function () {
  jobCache[A] = job(A, "succeeded", 2048);
  openHistoryConfirm("all");
  nextAnswer = {res: {ok: true}, json: {schema: "qwen_engine.web_jobs.v1",
                                        forgotten: 1, deleted_files: 0, skipped_live: 0}};
  confirmHistoryDeletion();
});

out.singleEntryForget = null;
sequence = sequence.then(function () {
  reset();
  jobCache[A] = job(A, "succeeded", 1500);
  openHistoryConfirm(A);
  var opened = snapshot();
  nextAnswer = {res: {ok: true}, json: {schema: "qwen_engine.web_jobs.v1",
                                        forgotten: A, deleted_files: 1}};
  confirmHistoryDeletion();
  return pump().then(function () {
    out.singleEntryForget = {opened: opened, after: snapshot()};
  });
});

run("refusedKeepsEverything", function () {
  jobCache[A] = job(A, "succeeded", 1500);
  openHistoryConfirm(A);
  $("history-confirm-files").checked = true;
  nextAnswer = {res: {ok: false},
                json: {schema: ERROR_SCHEMA, code: "job_unavailable",
                       message: "job is unknown or cannot perform this action"}};
  confirmHistoryDeletion();
});

run("countsMustBeSane", function () {
  jobCache[A] = job(A, "succeeded", 1500);
  openHistoryConfirm("all");
  nextAnswer = {res: {ok: true}, json: {schema: "qwen_engine.web_jobs.v1",
                                        forgotten: -2, deleted_files: 0, skipped_live: 0}};
  confirmHistoryDeletion();
});

out.sizeMath = (function () {
  return {bytes: formatBytes(512), kb: formatBytes(2048), mb: formatBytes(1572864),
          twoAndHalf: formatBytes(1572864 + 1048576),
          unknown: historyImageInfo([job(A, "succeeded", undefined)]),
          mixed: historyImageInfo([job(A, "succeeded", 1024), job(B, "succeeded", 3072),
                                   job(C, "failed", null)])};
})();

out.noImagesNoTick = (function () {
  reset();
  jobCache[B] = job(B, "failed", null);
  openHistoryConfirm("all");
  return snapshot();
})();

out.cancelledTargetIgnores = (function () {
  reset();
  jobCache[C] = job(C, "running", null);
  openHistoryConfirm(C);
  openHistoryConfirm("all");
  return snapshot();
})();

sequence.then(function () {
  process.stdout.write(JSON.stringify(out));
});
"""


def _run_history_harness(tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read("app.js")
    parts = "\n".join([_js_var(js, name) for name in HARNESS_VARS]
                      + [_js_function(js, fn) for fn in HARNESS_FNS])
    harness = tmp_path / "history_harness.js"
    harness.write_text(_HARNESS.replace("__PARTS__", parts), encoding="utf-8")
    run = subprocess.run([node, str(harness)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:2000]}"
    return json.loads(run.stdout)


def test_the_confirmation_names_entries_image_files_and_disk_size(tmp_path):
    out = _run_history_harness(tmp_path)
    opened = out["allUnticked"]
    assert opened["deleteAllDisabled"] is False       # a finished entry exists
    assert opened["confirmHidden"] is False
    assert "2 saved entries" in opened["question"]    # the queued job is not counted
    assert "Queued and running jobs are kept" in opened["question"]
    assert "1 image file" in opened["question"]
    assert "2.5 MiB" in opened["question"]             # roughly how much disk
    assert opened["filesChecked"] is False            # unticked by default
    assert opened["filesHidden"] is False             # the tick box is shown
    assert opened["yesText"] == "Forget 2 entries (files stay on disk)"
    assert opened["statusHidden"] is True


def test_confirming_deletes_over_the_exact_wire_and_reports_server_counts(tmp_path):
    out = _run_history_harness(tmp_path)
    done = out["confirmFilesDeletes"]
    assert done["calls"] == [{
        "url": "/api/v1/jobs/forget-all", "method": "POST",
        "contentType": "application/json", "ui": "1",
        "body": '{"delete_files":true}'}]
    assert "2 entries forgotten" in done["status"]
    assert "3 files deleted from disk" in done["status"]
    assert "1 running job kept" in done["status"]
    assert done["statusHidden"] is False
    assert done["confirmHidden"] is True              # the panel closes once sent
    assert done["cached"] == ["c" * 32]               # the live job's entry is kept
    assert done["refreshCalls"] == 1                  # the grid refreshes
    kept = out["confirmKeepsFiles"]
    assert json.loads(kept["calls"][0]["body"]) == {"delete_files": False}
    assert "image files kept on disk" in kept["status"]
    assert "from disk" not in kept["status"]          # nothing claimed, nothing deleted


def test_the_per_entry_forget_uses_the_same_panel_and_its_own_route(tmp_path):
    out = _run_history_harness(tmp_path)
    single = out["singleEntryForget"]
    opened = single["opened"]
    assert "Forget this history entry?" in opened["question"]
    assert "This entry holds 1 image file on disk, about 1.5 KiB." in opened["question"]
    assert opened["yesText"] == "Forget entry (image file stays on disk)"
    after = single["after"]
    assert after["calls"][0]["url"] == "/api/v1/jobs/" + "a" * 32 + "/forget"
    assert json.loads(after["calls"][0]["body"]) == {"delete_files": False}
    assert after["status"] == "Forgotten 1 entry, image file kept on disk."
    assert after["cached"] == []


def test_a_refused_or_insane_answer_deletes_nothing_locally(tmp_path):
    out = _run_history_harness(tmp_path)
    refused = out["refusedKeepsEverything"]
    assert "History was NOT deleted" in refused["status"]
    assert "(job_unavailable)" in refused["status"]
    assert refused["cached"] == ["a" * 32]            # cache untouched
    assert refused["refreshCalls"] == 1               # a refresh still shows the truth
    insane = out["countsMustBeSane"]
    assert "History was NOT deleted" in insane["status"]
    assert insane["cached"] == ["a" * 32]


def test_size_math_and_edge_targets(tmp_path):
    out = _run_history_harness(tmp_path)
    math = out["sizeMath"]
    assert math["bytes"] == "512 B"                    # the shared page formatter
    assert math["kb"] == "2.0 KiB"
    assert math["mb"] == "1.5 MiB"
    assert math["twoAndHalf"] == "2.5 MiB"
    assert math["unknown"] == {"files": 1, "bytes": -1}   # unknown sizes are never summed
    assert math["mixed"] == {"files": 2, "bytes": 4096}
    assert out["noImagesNoTick"]["filesHidden"] is True   # nothing to delete, no tick box
    assert "None of these entries has an image file" in out["noImagesNoTick"]["question"]
    assert out["cancelledTargetIgnores"]["confirmHidden"] is True  # a live job is no target
    assert out["emptyDisabled"]["deleteAllDisabled"] is True
