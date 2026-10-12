# Web interface pivot — in-flight senior-lead handoff

Agent: Codex/root; 2026-10-12 09:55 UTC. User requests continued build with
primary WEB interface (not planned Tkinter), similar to FlashML-org/FreeVideo.
Keep local Qwen-Image2.1 engine + CLI; no cloud/backend/model pivot.

Read vault README/Conventions, attempted memctl brief (root lock EROFS),
direct current STATE+repo checked. Registry vault absent. Approved normal
state-write/handoff work locally; don't retry remote push failures. Qwen
headless health96254 succeeded QWEN_READY today; previous SERVER500 no longer
current blocker. No server/model/config changes needed to regain response.

Root independent final JSON08993bc review completed: only seven decoder
boundaries widened; typed original probe green; guarded full772+1skip3.11 /
773pinned3.13 rc0. Qwen handoff's3.13 skip claim was an unguarded run; root
guarded pinned evidence has773passes. JSON accepted; cache7e0456d accepted
earlier. Repo baseline0a8c13e; user sandbox.yaml dirty preserved.

Root+sole peer design_audit inspected actual official FreeVideo web/studio.js,
view_navigation.js, library.js, freevideo.js, studio_queue/result_actions.
It's Comfy-hosted, not standalone. Borrow two-column progressive controls,
stable preview/status/progress, result stat cards and Creations history;
implement original standalone assets, no source/assets copying/Comfy/video/
Nodes/editor/LoRA/batch/auto-download dependencies. Reference:
https://github.com/FlashML-org/FreeVideo/blob/main/web/studio.js

Current sole Qwen writer51100, session-eeefdbe1-be3d-46dc-98ab-c2e6ed9656d9,
brief/tmp/v100-web-workspace-task.md W1 staticUI+design/PLAN only. Root owns
brief/admin notes, Qwen writes/commits all product code. Max3active participants
(root+Qwen+one read-only reviewer); design_audit now finished. W1 no backend
or generated images; UI must honestly show disconnected/Generate disabled.
Full64-bit seed kept decimalSTRING/BigInt, never JS Number; future API converts
validated decimal toPythonint and authoritative export emits numericJSON.
No invented speed/memory/quality figures. Mainguard remainsMLfree/offline.
Real browser CHROME_PATH=/usr/local/bin/chrome-headless-shell154; Node26.8.2
has builtinWebSocket, so CDP checks possible without dependency installation.

Next: visually/behaviorally rootreviewW1 with realChrome (desktop/mobile);
fix viaQwen as needed; W2 localstdlib server+same-origin bounded config/plan API
and packagedstatic assets (no auto-hashweightreads/MLimports); later job/event/
history plumbing with fake tests but no false real-generation readiness.
Runtime paths server-managed (browser cannot select arbitrary host files),
no change toexistingdsh-web3081 oruserportmapping. NewUIport/remoteaccess
needs separate explicit secure bind/auth/exposure implementation; don't claim
reachable outsidecontainer until tested. Engine tensor helper/codec and
encoderchild remain pending, then finalwholeprojectreview. Existing V100,
missingassets/torchvision/completeRAMbudget gates unchanged, no real load.

Commit this admin note and existing untracked Turbo assessment as found with
next docs milestone. No unrelated sandbox.yaml staging or asset mutations.

09:58 root early UI draft inspection: offload draft invented default/on/off
and attention candidate-a/b are NOT GenerationConfig enum values. Required
fix BEFORE accepting W1: read bounded config.py declarations and use exact
resident/streamed/offload and auto/sdpa/efficient/eager; Quantfp16/q8/q4.
Form snapshot keys match existing GenerationConfig except UI-only seed decimal
string (negative_prompt/true_cfg_scale/attention_backend/vae_tiled_decode).
No 'suggestion' semantics to silently normalize invalid enums later. Brief
/tmp/v100-web-workspace-task.md updated with exact values. If not seen by
current writer, send fresh focused fix task after its scoped milestone.
