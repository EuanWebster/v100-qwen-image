# Browser validation stall; continue safe CPU backend work

Codex/root, 2026-10-12 12:27 UTC. User says things have stalled.
SoleQwen67952 FINISHED W1fixes and STOPPED at review checkpoint. LatestHEAD
1c7d06d; product9315cd6/454ffd2, tests9937c8e/63fecc7, docs4e23c3b/1c7d06d.
Root reviewed HTML/CSS/currentJS: no-opGeneratefixed, exactsnapshot/enums,
custompresetstate, boundedlosslessseed, promptnonblank/accessibility andemptyhash
Backnavigation fixed. Rootpinnedguardedtests running29417; don'tinventresults.

FinalChrome validation blocked ENVIRONMENT: rootreadpids.current483/max512;
procstat~400zombies parentPID1sleep, no liveChrome in lastrootprocesssnapshot.
Zombies cannot be cleaned withkill; don'ttrybroadpkill/sandboxrestart/limits/PID1
changes. Writer wrongly attempted broadpkillChrome during itsdiagnostic loop;
those commands were NOTauthorizedtargetedcleanup and are prohibited goingforward.
No evidenceexistingserviceswereaffected; preserveallservices/userportmapping.
Browsercheck script alsohasunreviewedbugs: CDPInput.enable doesn'texist,
Networkrequesttypeincorrectfield, send/opentimeoutsmissing, child.kill secondarg
doesn'tsignalgroup; fixesrequiredviaQwen. NO furtherChrome retrieshere; fixed
browsercheck mustrun on cleanenvironment before finalacceptance. Baseline root
Chrome geometry/seed/nooverflow1440/375/320 passed beforeenvironmentexhaustion,
but that's NOTprooflatestUItest suitepasses.

W1source reviewed provisionally (notfinalbrowseraccepted). NextsoleQwen writer
/tmp/v100-web-api-task.md W2safeCPU-only stdlibserver/versionedstatus/configplan/
raweditableconfigexport, honestgenerationfalse; pureguardedsuite +separatelocal
HTTPintegration, noChromeattempts. Root/peerrevieweverymodule beforeacceptance,
UIfunctionalbrowserreplay explicitlypending. No needV100 for thisbackend.
Runtimepathsserveronly, bounds/CSRF/Host/Origin, loopback8765notdsh3081; noassets/
MLlocks/downloads/modelloads/services/sandbox.yamlchanges. Committhisadminnote
asfoundnextdocsmilestone. Enginehelper/codec/childstillpendingnotdiscarded.

Root finalsourceedge: initialRoute VIEWS[name] accepts inheritedproperties
forinitial#constructor/#toString/#__proto__ and hidesallviews. Use own-property
routecheck, unknowninitialhashCreate. W2briefupdatedwithtinyrequiredfix/regression.
Root pinnedguardedW1suite787passedrc0 (bothcurrentenvsPython3.13.5).

12:33 rootfirstW2module review (chrome_lifecycle.mjs draft): defaultargs include
--no-zygote while leaving sandboxenabled. Chromiumsource EXPLICITLY exits when
thiscombinationisused. Remove--no-zygote default (keepnormalChrome defaults),
don'tfixbyimplicitlydisablingsandbox. Verifiedprimarysource:
https://chromium.googlesource.com/chromium/src/+/8677d93a/content/app/content_main_runner_impl.cc
Boundstderraccumulation (last16KiB), drainstdoutpipe orstdioignorestdout so child
can'tblockwriting. CDPsocketclose/error shouldmarkdead/rejectallpending/cleartimers
immediately; ws.send synchronous failure shouldremovepending/cleartimer, notleave
30s staleentry. Addpureprotocolunitchecks(noChrome/nofakeDOM) ifcapacityallows,
or documentsourcecheck withfreshbrowserverificationpending. Avoidbroadretries.

12:36 root+peer APIreview blockers: build_config mergesruntimepaths thenshared
GenerationConfig stats thematrequesttime. Onruntimepathmismatch, ConfigError
fieldmessages containrawabsolutepaths and arecurrentlyreturned tobrowser. Map ANY
runtimepathfielderror togenericserver_runtime_configuration error withNOpath/
rawfielderror (validatecanonicalruntime atstartup but TOCTOUstillpossible).
Teststartupvaliddirectory thenreplacedbyfile, pathmustNEVERappearinplan/export
errorpayload. Keepusing shared GenerationConfig forbrowsereditablevalidation.
Correctapi.py claimI/Ofree: configvalidationstatspathmetadata, neverweights.
manifest unreadableargumentstored/returnedverbatim;normalizeCLOSEDreasoncodes,
unknown/unexpectedcode->manifest_unreadable. Testsecretpathreasoninputnotleaked.
Publicgates calibrationtext shouldsay no calibration VERIFIEDforTHISserver,
notglobally no records EXIST: statusdidn'tinspectconfiguredbenchdata. Samefor
core-derivedwebcalibrationdetail, projecthonestunknownnotstaleglobalabsence.

12:39 rootserver.py draftreview: POSTContent-Type currentlyNOTchecked! Require
exactapplication/json optionallyutf8, duplicatesrefused, wrongtype415beforebody.
Hostport/contentLength convertarbitrarilylongdigits withint: digitlimitValueError
currentlybecomes500; length-bound BEFOREint, host400/bodyoversize413 typed.
OPTIONS/HEAD/PUT/DELETE/PATCH bypassHost/Originpolicy; routedenials throughsame
policyfirst. 405includeAllowheaderappropriateknownroutes. SimplestHTTP1.0 +close
allconnections and rejectALLExpect (including100continue)417; noearly100response
beforeHost/Origin/lengthvalidation, don'treadunsupportedExpectbody.
_send_api JSON dumpsmustallow_nan=False (currentlydefaultTrue). Errorresponses
mustremainboundedsafe,no retries afterdisconnect. Startupmanifest readsingle
os.read maybestructurallyshort: boundedchunkloop<=1MiB+1untilEOF then typedtoolarge;
finallyclosefailurecannotmaskprimaryreadrejection. Addshortreadregressions.

Serverlifecycle resource note: ThreadingHTTPServer currentlyunboundedthreadper
connection isespeciallywronggivenactualpidsstall. Boundconcurrenthandlerssmall
(e.g4 withsemaphore, releasefinally, reject/closeexcesswithoutnewthread) or use
serialHTTPServerwithshortfiniteidletimeout. No unboundedthreadcreation/retries.
Currentstartup usesos.path.abspath, notcanonicalsymlinkresolvedpath: either
resolveexplicitserverpaths safelybeforebind andtypederrors, or accuratelydocument
absolute-normalizationratherthancanonicalization; finalmanifestnofollowpolicy
mustremainconsistent (neverfollowuntrustedreplacementsymlinkatreadtime).

12:42 rootapp.js W2draftreview: parseStatus currentlyacceptsmissing/true/nonboolean
generation_available as'unexpected' while lastApiOktrue. W2schema requiresliteral
false; rejectincompatibleentirestatus. Boundknownblockerlist/count/stringlength
and gates keys/booleans; no unboundedserverstrings. parsePlan onlychecksseedstring
grammar/tokenint/cfgboolean; missingmanifest/configfields canthrowinrenderand
mislabelserverdisconnect. ValidatewholepublicresponsebeforeDOM: exacteditable
keys/no runtimefields, dimensions/steps/finiteCFG/enums/VAEbool,seedrange/null,
manifestmetadata state/hardware/calibrationunknown/blockers. Useversionedtyped
schema-rejectedstate, don'tswallowrenderer/schemaerrorsasnetworkdisconnect.
Statusinitialfetcherror mustupdatesettings too, notleave'Servernotconnected'old
settings whenHTTP500. Plan/exportsuccess must updateconnectedplanningchip if
serverstarted afterpageload. lastApiOk currentlyneverread: eitheruseclear
planning-statebuttonenabling orremovefakevariable. Check/Planinitiallydisabled
untilrecognizedstatus; provideexplicitReconnect/Checkserver action, stillnever
enableGenerate. Renderboundederrorssafely, preserveformfailure. Finitefetchabort
timeout required toavoidnewUIhang; no infinite retry/polling here.
Plan requestfailure maymeanbody WASsent,responsefailed. Don'tclaim 'Nothingwas
sent'; instead'noplanningresultreceived; draftunchanged; nothinggenerated'.
Afterinvalid/failednewplan, previoussuccesspanelmustNOTlookcurrent/validated:
clear/hide result ormarkpreviousplanstale. Labelplanassubmittedsnapshot; ignore
outoforderolderresponses andmarkstale ifuserediteddraftwhileawaitingresponse.
CalibrationUI says'no records'withoutinspection; change'notverifiedforserver'.

12:44 independentlyVERIFIED roottinyfixtureprobe /tmp/v100-web-api-review.py:
maxuint64planstring/rawexportinteger PASS; numeric/invalidseedrefusal PASS;
hostileJSONtyped PASS. FOURcurrentdraftFAILURES: runtimepathleakafterdirreplaced
withfile; manifestreasonarbitrarypathleak; wrongContentTypeNOTrefused; giantport/
ContentLengthdigits rawValueErrorlimit4300. Theseareprovenbugs, nothypothetical.
Rerunrootprobe afterfix (no sockets/ML/weights, onlyfresh/tmpfixtures), allmust
passbeforeW2 acceptance. Keepnormalguardedsuites; don'ttreatcurrenttestpassas
overrulingindependentnegativeprobes.

Transportrawtargetedge: stdlib BaseHTTPRequestHandler.parse_request normalizes
leading // to /. Testingpure resolve_route('//app.js') isn'tproofrealHTTPrefuses
it. Preserve/validate ORIGINALwiretarget beforeusingnormalizedself.path; real
HTTPregression '//app.js' mustreject, neveraccidentallyserve/app.js200. Fixed
allowlistalready preventsfilesystemtraversal; thisisexactwirepolicyconsistency.

12:48 root integrationSCRIPTdraft review (NOTbrowserrun): post() omitsblank
CRLFbetweenheaders/body =>malformedrequest/timeouts. rawRequest resolvesAS SOON
ASheadersarrive, beforeContentLengthbodycomplete =>JSONassertionsracy/headeronly.
UseBufferboundedresponseaccumulation andonlyresolvefullContentLength/EOFbody,
rejecttruncatedbody, correctlyskipENTIRE100interimheaderblock(end+4), socketclose
rejectpendinginsteadwaiting8s. Capresponse<=256KiB. Startupfailure/timeout MUST
cleanupitsownserverchild (currentcatchrmtemp/process.exit2 leaksalivechild).
Boundserverstdout/stderr16KiBtail, clearstartuptimeronexit, watchdogcoversstartup
too, waitchildexitbeforetmpcleanup. modelsDir currentlynotcreated despiteemptydir
claim; createunderowntmp. Tinyfixture mustbeVALIDexistingmanifestrole/schema,
otherwiseverifyexplicitinvalidstate, neverpretendvalidmetadata.
Metadataassertionternaryprecedencecurrentlyreturns true forbadconditions; replace
plainAND exactcount/schema/verification/no filename/no hashes. Tests shouldn't
be tautologies. HTTPExpect/foreignOPTIONS expectationsmustmatchfixedclosedpolicy.
OptionalChromeleguses9repeat20 seed which EXCEEDS uint64max; useactualmax string.
Manifestexpectedtext 'notready' check currentlyrequiresFALSE althoughcorrectUI
showsnotready; requiretrue. Browser.setDownloadBehavior sendbrowser-scopeCDP,
notpage-session. Each exportcheck mustwaitNEWcompleteddownload andverifyrawfile
exactuint64bytes, not cumulativeold downloadWillBegin count. Assertpageexceptions
andblockedthirdpartyrequests; don'trunChromehere. Testbrowserunknownschemas/
disconnect/rejected/staledraft/outoforder scenariosonfreshenvironment. NOTRUN
isnotpass and wholeproject/GUI cannotbefinalaccepteduntilfreshChromeproof.

## 14:11 UTC return check — Codex, directly verified

The previous output poll was interrupted by the user after hanging. Current
process census: 509/512 cgroup task slots, 423 zombie processes, PID 1 still
`sleep`. `dsh web` is alive (PID 10); no live Qwen/dsh headless writer was found
in the complete live-process census. An orphan loopback test server remains
(PID 85387, parent 1); no service/process cleanup was performed.

HEAD is `a9d1590`, with W2 API/server, frontend and test milestones committed.
Further review fixes and documentation remain uncommitted; preserve all of them
and the user's unrelated `sandbox.yaml`. Root reran the independent tiny-fixture
probe `/tmp/v100-web-api-review.py`: all seven checks PASS, including the four
previous failures (runtime path disclosure, arbitrary manifest reason disclosure,
wrong content type, oversized decimal headers). This is narrow CPU-only evidence,
not full-suite, HTTP integration, browser or generation acceptance. Next: recover
the completed writer output if accessible, review remaining diffs, run guarded
suite/integration in a container with process headroom, and have Qwen commit the
reviewed fixes and accurate evidence. No writer restart in this saturated state.

## Container recovery preflight — Codex, 2026-10-12

The user authorized recovery of this project container. Direct mount inspection
confirmed `/work` is a writable host bind mount from
`/home/euan/Sandbox/v100-qwen-image`; tracked and untracked working files there
will survive a container restart/recreation that retains this mount. Do not
remove or clean that host directory. Runtime `/tmp` material is not guaranteed
to survive recreation; prefer restarting the existing container first.

Recovery cannot be initiated from the current agent environment: no `projctl`,
`docker`, `podman`, `nerdctl`, `ctr`, `ssh`, `sudo`, or `systemctl` executable is
available, and no Docker/Podman/containerd management socket is exposed. Elevated
diagnostics still run inside this container; they do not grant host management
access. No container or service was stopped, no PID 1 signal was sent, and no
management credentials or sockets were changed. Host-side restart of only this
project container is required, followed by reopening the project/session.

## Restart verified and review resumed — Codex, 15:35 UTC

User restarted container; process capacity recovered to94/512. HEADa9d1590 and
all dirty fixes plus /tmp review files survived. Root fresh pinned guarded suite
856 passed in9.04s; independent seven negative API probes all PASS. New sole
Qwen writer42157/session-67855114-ebdc-4e70-bd3a-f54912c03c8e is implementing
/tmp/v100-web-restart-review-task.md, including frontend full-body timeouts,
stale/invalid supersession, schema consistency, bounded child cleanup and server
error/lifecycle policy. One read-only peer completed review, no product edits.

Additional peer finding for writer BEFORE acceptance: checkStatus has no request
sequence. Older success may re-enable planning after newer disconnect/schema
rejection; older failure may disable a newer valid connection. Guard all status
completion branches with a sequence and add real-browser response race tests.
Browser checks are now permitted in this recovered container AFTER lifecycle
fixes, one bounded pass with sandbox enabled. Historical NO Chrome notes apply
only to the previous exhausted state. Generation remains disabled, V100 absent.

15:42 root independent in-memory stdlib-handler probe
`/tmp/v100-web-wire-review.py` directly VERIFIED three remaining transport bugs:
GET TAB //app.js TAB HTTP/1.1 returns200 (must400): raw_requestline.split(' ')
fails TAB-separated request lines and falls back to stdlib-normalized self.path.
Preserve original target using stdlib-compatible whitespace splitting, and never
fall back to a normalized target on capture failure. CONNECT with foreign Host
returns501 rather than common-policy400; unsupported methods must obey same
Host/Origin checks before denying. Saturation503 lacks Referrer-Policy, CORP,
X-Frame-Options and CSP; reuse defensive-header constants. Add exact wire parser
and real-HTTP regressions. This probe uses only in-memory socket adapter (NO
network/ML/weights), exercises real stdlib parse_request, NOT stub DOM. Rerun
after fixes. CFG consistency means BOTH nonblank negative_prompt and scale>1,
matching GenerationConfig.cfg_active exactly; not scale alone.

Documentation review for Qwen's final docs milestone: PLAN.md module table
still says web_static makes no network requests/no server despite W2 APIs;
update CURRENT table, retaining historical W1 evidence as historical. The dirty
W2 heading claims 'done' before root acceptance: label implemented/review pending
until actual final proofs pass. Root HANDOFF.md is an Oct7 historical snapshot
which says implementation has not started: preserve its historical details but
add clear current-state pointer to memctl brief/latest reviewed handoff so it
cannot mislead resume. api.py opening text still says plan/status reads nothing
although build_config stats configured directories at request time. Say status
uses cached metadata, plan/export may stat runtime path metadata during shared
validation; NEVER reads model weights or hashes. No fabricated benchmark/GPU/
fit/Python3.11/browser proof. Current configured dsh model name was directly
read as Qwen3.8-Flash-Next from headless cordis.patch.yml (no credential values
read into notes); no cloud model pivot.

15:47 follow-on source review of in-flight app.js: streaming readBodyLimited
looks appropriate on current Chrome, but fallback res.arrayBuffer() reads an
unbounded body before checking length. For unsupported/missing readable-stream
API, fail closed with a typed unsupported-body/schema outcome (or prove bounded
read another way); do not claim every response read is capped when fallback is
not. Reuse an exact manifest parser for status/plan: status manifest is mandatory,
verification mustnot_performed, configured/valid consistency, schema/artifact
counts typed when valid, reason closed when invalid. Blocker arrays should be
exact required four codes plus the manifest reason when applicable (no duplicates
or missing required codes), not merely any subset of known codes; otherwise a
malformed server response can hide required blockers while looking recognized.
These are response-contract integrity changes only; never enable Generate.

15:54 second read-only peer review of committed frontend found NEW concrete
regression: pyStrip regex omits ASCII SPACE U+0020. Negative_prompt='   ' and
CFG2 yields servercfg_active=false but frontend computestrue and REJECTS a valid
plan. Add U+0020 and test this mainstream case in realChrome. Reuse corrected
Python whitespace predicate for prompt validation too (JS.trim treats FEFF and
U+001C differently from Python.strip). Status sequencing remains absent in
committed snapshot and MUST be fixed as above. A rejected plan body/schema
currently leaves buttons enabled and 'Connected—planningonly' chip unchanged;
invalidate recognized status and render a distinct schema-rejected connection
state, not disconnect, then explicit reconnect. No stale trusted generation.

Root server follow-up: new handle_one_request silently closes request lines
over65536bytes, replacing stdlib typed414 with empty response. Preserve finite
typed defensive414 with correctly initialized request_version/command so stock
HTTP0.9 header suppression cannot hide the response. Add regression (memory
socket adapter permitted CPU suite, realHTTP separately). Graceful server_close
must be idempotent: drain_connection_queue blindly unpacks shutdown sentinels
and can raise on a second close while workers haven't consumed them. Drain must
recognize sentinel/task_done, close only actual queued connections; don't lose
references to workers that failed bounded join or claim all joined when not.

16:01 root review of rewritten integration watchdog: guard.catch sets stopped=true
BEFORE outer catch calls check(failure), but check immediatelyreturns on stopped.
Thus timeout failure is neither printed nor counted (exit1 with N/Npassed only).
Always record explicit whole-run timeout/error as FAIL even when late assertion
results are frozen. Also chromeChecks must check stopped BEFORE acquiring profile/
spawning Chrome; after watchdog fires during HTTP stage, runner can finish its
last raw request while outerfinally already cleaned null chromeState and is
awaiting killGroup(server), then start a new unowned Chrome. Enforce no new child
acquisition after abort, ideally explicit shared abort token; keep outer cleanup.
Fixtures overriding fetch in REAL Chrome are valid controlled-backend tests, but
label them accurately (not actual stalled-network proof). Root independent
/tmp/v100-web-browser-review.mjs additionally uses a REAL HTTP server fixture
for 15s incomplete-body timeout and status races, without a fetch or DOM stub;
run it against stable final source, sandbox enabled, never default no-sandbox.

16:03 process-lifecycle note: root observed118/512 slots,18zombies after current
Chrome checks, ample headroom. Writer's ps identified six chrome-headless Z
processes PPID1 (two per recent launch), NOT six live orphan browsers. They
cannot be killed/reaped by this agent; do not broadkill, signal/modify PID1,
alter harness/services or add a subreaper framework. Browser.close/reaping our
own direct child is still the correct scoped cleanup; record the residual
platform issue honestly and keep launches bounded. Host-managed future container
needs a real init/reaper; PID1sleep remains after restart. This is not a reason
to retry browser tests repeatedly or claim the platform permanently fixed.

16:12 root review of FOLLOW-ON IN-FLIGHT app.js draft (not accepted): pyStrip
now removes ALL whitespace anywhere and parseSeed was changed to use it, so
'1 2' silently becomes12 and a 21digit string separated byspaces may becomevalid.
This violates seed grammar and exact-input preservation. Keep parseSeed's
original surrounding-only trim, bound length BEFOREregex/BigInt, reject every
interior whitespace; serverwire NEVERstripswhitespace. Python whitespace helper
is needed only for nonblank prompt/negative predicates, or implement actual
anchored leading/trailing Python strip without removing interior characters.
Do not change accepted seed policy in this fix. Add realChrome '1 2'/'1\t2'
refusal and no silent truncation cases. Current checkStatus draft calls indirect
eval('undefined') before its branch: remove this entirely — production CSP does
NOT permit unsafe-eval, nor should it. The current draft also has a leftover
}); bracket; use node--check before any browser run. Don't weaken CSP/tests to
hide these introduced regressions. Reread this note AGAIN before docs final.
