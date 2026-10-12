# W1 review in flight — root correction and next milestone

Codex/root, 2026-10-12 10:06 UTC. Baseline W1 product89041f9/docs25aaf15.
Root independently ran realChrome CDP at1440/375/320: nooverflow/pageexceptions,
exactmaxuint64string, correctsquare/landscapegeometry, routesvisible; stale
presetconfirmed (Landscape stayspressed after dimensionschangedto1024square).
Root/peerreview alsofound ready:true enablesno-opGenerate, unboundedBigInt,
missinginvalidARIA, routefocus/skip-linksemantics, crampedseedcontrol.

SoleQwenwriter67952 session-c66c3ad7-919d-4191-aa71-0e4a10ef509d executing
/tmp/v100-web-w1-fixes-task.md. RemoveW1speculativecapabilityprobeentirely,
alwaysdisabledwithoutadapter/jobAPI, focused fixes/reproducibleChrome script,
fullguarded3.11+pinned3.13 thenproduct/docscommits noamend and STOP.

ROOT CORRECTION: initial review brief incorrectly said GenerationConfig
allows empty prompt. Actual config.py215 checks nonempty trimmed string.
Brief corrected10:06. W1 lightweight validation must reject blank prompt
with accessibleerror afteredit/check/submit, avoidinitialintrusivealert,
neverautofillprompt. Followactualconfigschema, don'trepeatrootincorrectclaim.
Include this admin note as found in nextdocscommit;don'ttouchsandbox.yaml.

Next W2 afterrootacceptance: /tmp/v100-web-api-task.md, stdlibstaticserver +
authoritative config/plan and rawconfigexport. /api/status, /api/plan; no old
/api/v1/capability; actualproductiongeneration_availablefalse. Defaultloopback
127.0.0.1:8765 (notdsh3081), Host/Origin/CSRF/boundedrequestguards, puremaintests
separatelocalHTTP+Chromeintegration. No weights/hash/ML/statusload. Container
localhostnotautomaticallyhost/Tailscalereachable. Runtimepathsserveronly.
Enginehelper/codec/childremaining notdiscarded. KeepV100/asset/runtime/RAM gates.

11:05 rootlivecode review: routeOnHashchange currently ignores emptyhash,
so History navigation from initial nohash then browserBack does NOT restore
Create. Treat emptyhash asCreate, while #main/otherunknownhash preserveview.
Add realChrome back/forward regression (notjustassignknownhashes).
Prompt correction above STILLapplies currentdraftvalidateForm lacks prompt.
Please read this note beforefinal milestone, implementthese two small fixes
in currentreview-fixcommit and update browserchecks. Default .venv nowobserved
Python3.13.5, asispinnedreviewenv; measureactualcounts, rootold3.11claimstale.

Additionalrootlosslessnessfinding: HTMLmaxlength20 truncatesa21digitpaste to
a DIFFERENT potentiallyvalid20digitseed (e.g.100000000000000000000 becomes
10000000000000000000). Avoidsilentseedchanges: removeHTMLmaxlength entirely,
keepparseSeed length<=20 BEFOREregex/BigInt, inlineerrorforoversizeinput.
Alternativelyexplicitlypreventoversizepasteandshowerrorwithouttruncation, but
removingmaxlength is simplest. Testactualbrowser Input.insertText/pastelike
entry of21digits, expect originalinput retained plusinvalid, notvalidtruncation.
Don'ttradehonestuint64inputfordeveloper-orientedmaxlengthtexttests.
