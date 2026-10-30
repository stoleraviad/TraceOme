# Bio Neb Observable Agentic Backend

The canonical language for the researcher-visible run, its scientific evidence, and the human and
agent decisions that shape it.

## Run and observation

**Run**:
A researcher-visible execution of one scientific request, described by one Journey.
_Avoid_: Thread, workflow instance

**Journey**:
The append-only semantic event record that determines what happened in a Run and what the product
may derive from it. It is the source of the researcher's visible path, decisions and evidence history.
_Avoid_: Trace database, LangGraph thread

**Path fact** (0032):
A typed fact on the Journey's meaningful path that supports Reconstruction, a later decision,
grounding by a judge, or a human request or answer. Membership is a test, not a list.
_Avoid_: Provider contact, diagnostic trace, heartbeat

**Grounding** (0032):
The stated basis of a Path fact, one of five tokens: `verified`, `tool_succeeded`,
`agent_composed`, `ambiguous`, `human` (prose: "grounded human"). It distinguishes evidential
support from the kind of fact recorded; human authority alone does not establish scientific truth.
_Avoid_: Confidence score, approval as proof, human-grounded

**Contact journal** (0033):
The retained mechanical record of actual provider, source, Tool and model contacts, correlated to
logical requests, Attempts and items with receipts, effective policy and resource identities. One
append-only journal belongs to each coordinator Pass and closes when that processing interval ends.
_Avoid_: Journey, Run ledger, live heartbeat stream, accounting

**Contact review** (0033):
The projection of the Contact journal over every retained Pass, including one that ended without
an outcome; the source of resource counts for tests and the Run ledger.
_Avoid_: Journey projection, run status, accounting

**Run ledger** (0033, 0037):
The DEV inventory of retained resources keyed by kind, location and identity, derived from APP
results and the Contact review through list-only provider calls. Its removal columns are written
by the owner after a manual act; no tool writes them.
_Avoid_: Contact journal, third authority, DEV ledger, cleanup tool

**Pass** (0032, 0031):
One active processing interval of the Run coordinator. It retains meaningful facts as work proceeds
and closes its Contact journal once when the interval ends. An item's pending human question ends
that item's Attempt while independent authorized work can continue in the same Pass. A later lawful
instruction or exact answer starts another interval for eligible affected work.
_Avoid_: Attempt, human-wait process, Run completion

**Journey bundle**:
A Journey JSONL record together with the immutable artifacts it references under the Run's external
root.
_Avoid_: Log file, checkpoint database

**Attempt**:
One transient invocation of shared Agent work for a scoped assignment, with its own context,
Tools, policy, results and question identities. Clarification ends the Attempt; a later answer can
start a new one without replaying completed work. A provider retry within one logical request is
not another Attempt. Several independent Attempts may belong to one coordinator Pass.
_Avoid_: Journey, physical request, retry, Pass

**Journey projection**:
A typed read model derived from Journey evidence for application decisions and researcher-facing
presentation. CLI and UI consume the same projection; it never runs work or creates scientific truth.
_Avoid_: Framework state, mutable status store, UI-owned state

**Lifecycle observation**:
A diagnostic record that an agent, tool, or adapter activity started, remained live, changed
material state, succeeded, or failed. It does not prove scientific validity by itself.
_Avoid_: Verified result

**Slice checkpoint**:
A successful boundary showing that the Run reached the frontier declared by the active ticket while
the wider product objective remains incomplete.
_Avoid_: Objective completion, missing-feature failure

**APP initialization**:
The future Run-independent setup boundary that guides registration of per-account values (the
model credential's value, the Nebius service account, a Run's bucket) and validates their
availability before the APP accepts a Run. Model and provider selections already have build
defaults, so a Run begins on them without it. It never stores credential values or places them in
Journey evidence.
_Avoid_: Pre-Parabricks readiness, Run registration

**Pre-Parabricks readiness**:
The per-Run boundary after Execution authority and before execution preparation or resource
mutation that waits for Nebius CLI-owned interactive authentication when needed and validates the
exact Nebius scope and unique operational topology. It does not select global defaults or own
login.
_Avoid_: APP initialization, execution composition gate, login flow

**Execution composition gate**:
The automatic contact-free structural check inside the real deterministic execution command after
final Execution HITL and before authority consumption, Journey execution append, credential
resolution, or provider contact. `bio_neb.parabricks.plan.plan_execution` derives one typed plan
from approved facts and retained state. Its refusal families are verification policy, Nebius
context drift, lineage, and paired FASTQ roles; source and handoff relationships are constructed
by the plan's types and layout. Runtime alternatives and ordered fallbacks come from the approved
proposal's capability snapshot. Provider create safety belongs to deterministic names and the
provider's exists-precheck. The gate does not predict provider readiness or replace live validation.
_Avoid_: Provider dry run, offline acceptance Run

**Controller runtime**:
The one supported Linux development container in which the APP, Journey access, Nebius CLI, local
paths, credentials, and provider-facing process mechanics remain for the lifetime of a Run. Its
machine-specific locations are named by one policy (`bio_neb.locations`). A host may create or
recreate the container, but the host is not part of Run meaning.
_Avoid_: Laptop OS, host shell hop, worker container

**Walking Run**:
One fresh live Run advanced cumulatively through ticket-scoped provider gates so each boundary is
visible before the next consequential stage. A later acceptance Run uses wholly fresh authority and
resources; the walking Run is evidence, never reusable execution authority.
_Avoid_: Resumed failed Run, production workflow

## Agentic work

**Agent role**:
A named planning, research, validation, routing, judging, or synthesis responsibility with its own
instructions, typed inputs and outputs, and lawful tool access.
_Avoid_: Versioned agent, ticket agent

**Investigation Planner**:
The Agent role that turns attributed User assertions and supporting evidence into a normalized,
collection-level Investigation proposal. It does not schedule Work items or grant scientific truth.
_Avoid_: Run coordinator, investigator, workflow planner

**Paper Reader** (0046):
The Agent role that reads one retrieved paper for a researcher's question with reading Tools
(outline, search, read, table rows) and answers by pointers into the paper. Code copies what the
pointers name; the model's reasons are evidence to verify, not scientific fact. In a Run it reads
the paper a plan cites, in planning before the reference work (#383 A2); the development tool's
`answer` command (`tools/paper_retrieval`) also runs it over a downloaded paper.
_Avoid_: paper agent, paper search, extractor

**Reading item** (0046):
One unit of a Paper Reader's work, run as its own Attempt: the paper item proposes the areas the
question needs, and each area, and each id the question lists that code did not find, is its own
item answered by pointers. Reading items come before Investigation planning and feed it proposed
targets.
_Avoid_: chunk, page, Work item

**Run paper** (#420):
A paper that entered a Run: supplied by the researcher as a file, or cited in the researcher's
words and retrieved by the planning Agent. Each of its files is retained once by digest, beside
the normalized text every reading cites, and one `PaperRecorded` names it; its `paper_id` is
derived from its files' digests. The Run's papers are the only paper corpus; nothing in the
repository provides one. A paper that cannot be read or retrieved asks the researcher.
_Avoid_: corpus paper, preset paper, gold paper

**Effective invocation policy**:
The resolved secret-free settings governing one Agent loop or Tool invocation, including inherited
defaults and explicit overrides. Its recorded meaning belongs to that invocation; later amendments
govern subsequent work.
_Avoid_: Current defaults, execution approval

**Effective failure policy** (0031):
The resolved, retained failure settings governing one unit of work, derived from shared defaults
and its explicit applicable work or invocation overrides. Contacts and failure evidence retain the
settings that governed them; later amendments apply to subsequent work.
_Avoid_: resolved policy, policy snapshot, retry config

**Model endpoint**:
The selected Nebius model-inference offering contacted by an Agent role through an Adapter. An
OpenAI-compatible request format or SDK is Adapter mechanics and does not name the provider or the
domain responsibility.
_Avoid_: OpenAI model, generic provider

**Model turn**:
One exchange with a Model endpoint as the loop reads it: the reply's calls and their arguments,
how the reply ended (a cut-off reply keeps every call it held) and the retained transcript. One
reader reads every model output the same way; arguments written inside one Markdown fence are
stripped, accepted and recorded as fenced. Every correction the model reads in the next turn is
written by the failure module's retry rendering in the Model endpoint policy's wording.
_Avoid_: response parser, output handler

**Model endpoint policy**:
The one seam holding everything specific to a model: its request defaults, the settings it accepts,
how each setting is carried on the wire, how a Tool's schema is shown to it, its correction
wording, the injection patterns it switches on, where it is served, the name of its credential
variable and its own Run defaults (fault allowances and Tool choice). Each listed model's policy is
one entry of the build's model registry, which marks one default entry; an unlisted model has
none and is refused. Policy config overrides the defaults and the Tool choice by scope (global, then Agent
role, stage or Tool); the result resolves once into the Effective invocation policy. Each Attempt
binds its model's policy once and its role, assembly and adapter read that bound policy. Roles,
Tools, stages, budgets and guards stay with the shared Agent work.
_Avoid_: model profile, per-model branch, fallback model

**Injection slot** (0048):
The one place in the shared Agent work where registered patterns speak to the model mid-Attempt. A
pattern names its trigger and its message; the Model endpoint policy switches it on and policy config
tunes it by scope. Each injected message is a name-tagged user message in the pending request, kept
in history and in that request's record. A pattern only speaks: a cap is a notice, never a stop.
_Avoid_: middleware, request-only hint, nudge hook

**Tool**:
A curated model-callable operation accepting validated arguments and returning a typed, retained
outcome correlated to its exact call. Its domain meaning determines which results become Path facts;
mechanical execution alone grants neither scientific truth nor authority.
_Avoid_: capability inventory entry, scientific fact

**Repeated call** (0044):
Within an Attempt, a lawful Tool call with the same Tool and arguments as an earlier admitted call
when nothing meaningful has changed since. The loop guard stops it before contact: it adds no
evidence, and enough of them stop the Attempt for the researcher's review. A changed result lets the
call run again.
_Avoid_: retry, duplicate fact

**Turn routing** (0044):
What the loop does with one reply before any of its calls runs: copies of a call collapse onto it,
each unoffered call is refused on its own, an output sent beside a call that runs is refused (an
output must come alone and after its evidence), a cut-off reply's last call (the partial call:
nothing shows it ended) is dropped, and the kept calls are admitted once each in dependency order.
The kept reply holds only those calls; the raw reply stays in the Journey.
_Avoid_: response filter, call splitter

**Turn fault** (0044):
A fault of one Model turn that the failure owner counts under its key: a refused call, a text ending,
copies, unoffered calls, an output not alone, a cut-off, a repeated call or a resend after a timeout.
Each has an allowance; the turn's one note names its reply faults with their counts. A resend is a
short failure's one further send of the same request; a **stall** (no reply within the stall limit)
is not one: it ends the request, and the next request carries the stall note (#443).
_Avoid_: model error, retry reason

**Adapter**:
The external, provider, filesystem, or process mechanics behind a Tool or deterministic execution
operation. It does not own workflow meaning.
_Avoid_: Controller, authority

**Run coordinator**:
The application-code Run coordinator that enforces the curated end-to-end path and gates,
schedules eligible operations and Agent assignments, and routes corrections using checked Journey
meaning. It is a responsibility of the application, not a separate Agent or authority store.
_Avoid_: model-owned workflow, Investigation Planner, Journey, scientific authority

**Variant target**:
An assembly-bound normalized variant identity comprising contig, position, reference allele,
alternate allele, and exact grounding evidence. An rsID or free-text variant remains a User
assertion until it resolves to this identity.
_Avoid_: rsID, expected variant, observed variant

**Work item**:
A stable collection member for one normalized scientific target or question, owning its
sample-scoped observations, evidence, assessments, and current disposition. One Work item may own
one or more Sample observations. A request whose source lists several alternate alleles has one
candidate Work item per allele; the researcher's approval selects among them.
_Avoid_: Fixed slot, sample, variant record, stage number

**Candidate Work item**:
A target or question discovered from User assertions or retained evidence that has not yet received
Investigation authority. It may be normalized and deduplicated, but it cannot execute Tools until a
mandatory Review approves its exact actions within a collection revision.
_Avoid_: Authorized target, pending Tool call

**Work-item disposition**:
The retained state and outcome of one Work item, including lawful uncertainty, partial results or
pending approval. It is distinct from permission for other work to continue and from Run completion.
_Avoid_: Whole-Run status, scientific failure

**Investigation revision**:
An immutable, authority-scoped collection snapshot that carries unchanged Work items forward and
records explicit additions, supersessions, deferrals, removals, or search-policy changes.
_Avoid_: Mutable target list, replacement query

## Evidence and decisions

**User assertion**:
A value or scientific claim supplied through the prompt or HITL. It is authoritative as user input
but requires the appropriate evidence before becoming an Admitted fact.
_Avoid_: Verified fact

**Identifier rule**:
The one named, domain-owned format matcher for an identifier kind (rsID, sample identifier, NCBI
taxon id, assembly accession, output kind). It checks a typed value against its quote and never
reads a value out of the quote or any free text (#472); a matched value stays a User assertion, and
identity beyond format belongs to the source.
_Avoid_: Canonicalizer, normalizer

**Membership check** (0049):
The one check code makes on a model choice: the record it cites is one its source returned, and the
quote it cites holds the value in the researcher's or the paper's words. It establishes neither truth
nor entailment.
_Avoid_: Validation (of a judgment), verdict

**Retrieved source**:
An identified representation obtained from an external source, retained with retrieval and artifact
lineage.
_Avoid_: Search result fact

**Variant source** (#366, 0049):
A public database a Run's rsID lookups read: NCBI dbSNP or the European Variation Archive (EVA).
Every Run is offered both; Tool metadata says which suits which organism, the model chooses, and a
source with no record leads it to try the other. Each answer is a Retrieved source named by its
source's authority.
_Avoid_: Variant database, dbSNP (as every Run's source)

**Stated position** (#369):
Where a request places its target: a sequence as the request names it, a one-based position and,
when stated, its alleles, with the genome version its source states and the line that states it. It
is a typed User assertion, never read out of prose. Its lookup places it on the Run's assembly, checks
its REF against the Run's reference, and reads what the Variant source lists there; a position no
source lists is checked as stated, with no rsID. A position stated on another version is compared
only with the source's own placement on that version; no coordinates are mapped between versions.
_Avoid_: Coordinates in the request text, lifted position

**Placed position** (#367):
Where placement put a target on the Run's assembly: the step that placed it (its stated version is the
Run's, a flank move, NCBI's Variation service, or G15 on a gene and change), the written version name
that matched, and the Stated position it came from, kept beside it unchanged. It is never written or
shown as a Stated position on the Run's version. A placed target is looked up at its placed position;
a target no step placed keeps its reason, and with no rsID it is recorded as not checked.
_Avoid_: Stated position on the Run's accession, lifted position

**Sample observation**:
An immutable, sample-scoped result of asking whether one Variant target is represented in one
attested VCF, with each record's call quality recorded as written. Neither a missing record nor
uncertainty establishes coverage or biological absence.
_Avoid_: VCF fact, absence finding, Lifecycle observation

**Deterministic lookup** (0047):
The automatic, model-free VCF observation of one approved Work item; its outcome alone decides the
item's disposition and admission.
_Avoid_: Agent lookup, VCF fact

**VCF table** (0047):
One row per approved Work item after the lookups: its lookup outcome, its call and call quality as
the VCF wrote them, its fixed window of the Run's prepared VCF read with no model, its coverage and
the cases across requested SNPs that touch it. It extends the Collection view.
_Avoid_: second table, raw VCF window

**Table reading** (0047):
The table reader's Agent assignment over one chunk of the VCF table, ending at one reading per row:
no concern, or a concern with its codes and a short note. A reading reports; it never changes a
lookup, admission or scientific status.
_Avoid_: item query, per-item Agent, Review (a Review is a human gate)

**Item evidence** (0047):
One Work item's own record: its Deterministic lookup outcome and the typed fields other steps fill,
its window, its reading and its coverage; a field no tool filled reads "not measured". It grants no
scientific status.
_Avoid_: item fact, coverage claim

**Candidate evidence**:
A typed proposed scientific claim or relevant passage, attributed to its originating assertion and
linked to exact Retrieved-source or Sample-observation lineage, that awaits assessment or admission.
The underlying source or observation remains distinct.
_Avoid_: User assertion, final fact, raw Tool result

**Evidence assessment**:
An immutable, authority-scoped evaluation of one Candidate evidence item against an exact basis,
recording claim scope, verdict, rationale, assessor identity, and supersession. It does not admit
the claim.
_Avoid_: Admitted fact, human selection, model opinion

**Admitted fact**:
A claim-scoped Candidate evidence item accepted by claim-specific authority for downstream
reporting, with exact lineage to sources, observations, assessments, and decisions. Human selection
or approval alone cannot grant this status.
_Avoid_: Model statement, User assertion, Sample observation, approved-by-user fact

**Settled**:
A requirement that reached its end conclusion: resolved (a positive value) or rejected by the
researcher. Work that consumes a set of requirements starts only when all are settled.
_Avoid_: Done, closed

**Requirement value**:
The current settled value of one requirement, its source (the researcher's request, a Tool result or
a Review answer) and its status (asserted or verified). It is not an Admitted fact.
_Avoid_: Admitted requirement fact, accepted intent

**Requirement resolution**:
A Journey projection that preserves the original decomposition while deriving each requirement's
Requirement value from attributed assertions, Tool evidence, Review responses and contradictions,
keeping each value's status (asserted or verified). Neither status makes it an Admitted fact. Only
consequence-bearing requirements that are not Settled prevent proposal compilation.
_Avoid_: Mutable plan, planner-value viability

**Execution requirement resolution**:
The exact, consequence-only snapshot of the resolved requirements' Requirement values, with their
status and lineage, bound into an Execution proposal. It grants no Admitted-fact status. Package
compilation binds it through the compiled proposal and scientific continuation must match it;
post-execution scientific investigation items remain outside it.
_Avoid_: Current-value lookup, full information plan authority

**Scientific investigation handoff**:
A typed, lineage-bound collection of known variants, literature claims, hypotheses, discovery
requests, and open questions carried across Execution as input to post-run planning, with the
Research goal and each item's Targets in the order the researcher wrote them. It is neither an
Investigation input nor an Investigation proposal and grants no Tool or scientific authority.
_Avoid_: Investigation proposal, expected-variant execution gate, demo slots

**Research goal**:
What the researcher asks the Run to find out, in the request's own words (the plan's quote). Every
Investigation member reads it as its original intent; no single target stands in for it.
_Avoid_: First item, original intent as one rsID

**Target**:
One thing a scientific investigation item asks about, in the researcher's own words: the request's
quote for it, an rsID a researcher's answer states, or an item of the registered request file the plan
cites. Planning owns the Run's target list, settled before the Execution proposal; text the planning
model wrote is never a Target.
_Avoid_: Planner paraphrase, file override, Variant target (the normalized identity)

**Investigation input**:
The minimum post-run scientific binding of an immutable VCF identity, represented sample identity,
and assembly. A VCF index is an access condition; optional alignment artifacts remain separate
evidence for later questions such as coverage.
_Avoid_: Scientific continuation packet, upstream execution choreography

**Investigation**:
The researcher-visible post-run inquiry that turns an Investigation input and scientific question
into collection-shaped Work-item outcomes, evidence, and decisions.
_Avoid_: Workflow, graph, report

**Investigation proposal**:
An immutable collection-level description of normalized Work items, their grounding evidence,
intended lawful post-VCF actions, and unresolved choices, presented before VCF observation begins.
_Avoid_: Prompt, draft plan, Execution proposal

**Investigation review**:
The Review junction bound to one exact Investigation proposal revision, allowing independently
selected Work-item approvals while other members await changes. Several decisions may be made
together; changed, added, and explicitly reopened members become User assertions requiring
verification and a superseding proposal, while unchanged members carry forward exactly.
_Avoid_: Execution HITL, scientific admission, free-text correction

**Issue item** (0031):
A requested target whose check ended with an issue (not found, other assembly only, conflict). Its issue is keyed with its verdict and evidence and answered on its own:
the researcher approves the finding as checked but not verified, amends the request so only that
target is checked again, or rejects it, which ends the Run. The report lists it as such.
_Avoid_: Skipped variant, failed variant, unresolved request identity

**Collection view** (0039):
Every requested target of an Investigation shown together, as the researcher asked it and as its
sources and lookups read it, with each cross-target case that is not simple (two requests resolving
to one rsID, two rsIDs at one locus, sources that disagree, several alternate alleles, a target with
no Work item, an overlapping record, mixed outcomes, a trimmed match). A case reports; it never
admits a claim.
_Avoid_: Summary verdict, combined answer

**Investigation authority**:
The exact approval of selected Work-item actions within an immutable Investigation proposal
revision. It authorizes only those post-VCF actions and grants neither Execution authority nor
scientific truth.
_Avoid_: Approval text, Execution authority, Admitted fact

**Scientific conflict**:
Correctly linked evidence that supports incompatible claim-scoped conclusions for one Work item.
It is a lawful disposition requiring assessment, more evidence, HITL, or explicit unresolved
treatment rather than a structural Journey failure.
_Avoid_: Projection error, malformed evidence, automatic Run failure

**Designed stop** (0031):
A scoped point where a Review junction requires a human decision with declared legal actions and
consequences. A lawful answer continues affected work from checked retained state with its original
grounding. Independent authorized work need not stop; DEV milestones are not Designed stops.
_Avoid_: HITL pair, stop class, global pause, checkpoint

**Review junction** (0031):
The mechanism at a designed stop: a Journey-owned request declaring its Junction legal actions,
answered by one authority fact grounded human that rests on the request. Each junction resolves
only its own stop; an identical replay is a no-op and a different answer is a conflict. Named
instances are "<domain> review" (CPU preparation review, failure review, Investigation review).
_Avoid_: UI dialog state, LangGraph interrupt, HITL junction, ask (0031's route to the stop)

**Retry preset** (0031):
A researcher's `retry`, given ahead of time for one Agent role's failure reviews of named failure
classes. It is bounded by a count of uses within the Run. Each failure review it covers opens and is
answered by the preset's own author in the review's ordinary record. A later preset for the role
replaces it, and an invariant is never preset.
_Avoid_: auto-retry, blind retry, retry policy

**Junction legal actions** (0031):
The choices a Review junction explicitly permits, including amendments only to fields the plan
declares amendable (default: none). A human answer grants the stated authority; an amendment's
content is a User assertion carried in the answer payload and routed to verification. "Legal
actions" is the one term for the choices, on every surface and in every page.
_Avoid_: Unrestricted approval, arbitrary edit, legal answers, legal choices

**Execution proposal**:
A complete immutable description of the intended Parabricks consequences, compiled only from facts
whose required validation has succeeded.
_Avoid_: Draft plan, prompt

**Capability snapshot**:
An immutable set of exact execution recipes, resources, images, and tool identities available to
Planning. Its content identity is derived by the application from the selected claims. It is not a
Retrieved source and does not prove that a model read or reasoned over scientific evidence.
_Avoid_: Research source, manual digest assertion

**Execution authority**:
An Execution proposal that received exact HITL approval and may be consumed by the deterministic
execution Adapter.
_Avoid_: Approval text, proposed command

**Finalization basis**:
An immutable preapproval snapshot of scientific dispositions, admitted facts and their supporting
evidence, together with actual actions, issues, decisions, rejections, exclusions and limitations.
It may account for requests that never became normalized Work items without admitting a finding.
_Avoid_: draft report, current mutable view, Report basis

**Finalization review**:
The mandatory Review junction bound to one exact Finalization basis, offering approve, correct, or
stop. Correction creates a new basis, approval creates a Report basis, and stop grants no report
authority.
_Avoid_: Investigation review, report editing, approval text

**Report basis**:
The exactly approved Finalization basis and sole content basis for a Grounded report or optional
Explanatory synthesis. Scientific membership and accounting for work, rejection or uncertainty keep
their distinct classifications, and each Qualification keeps the ids of what it concerns.
_Avoid_: draft report, current evidence view, Finalization basis

**Qualification**:
One limitation, conflict or gap a Finalization basis records: its sentence once, with the ids of the
Work items and requests whose state or evidence stated it. One stated by none, or concerning every
looked-up row, is the Run's own.
_Avoid_: caveat, warning, finding

**Grounded report**:
A durable researcher-facing answer derived only from one Report basis, with exact evidence pointers
and visible limitations. It creates no new scientific fact.
_Avoid_: Evidence package, Explanatory synthesis, Journey projection

**Explanatory synthesis**:
A bounded prose explanation over one Report basis, requested by default for every published report, with read-only retrieval of retained
evidence within that approved basis. It is non-authoritative, performs no new scientific investigation,
and may not add facts, evidence, or scientific scope.
_Avoid_: Grounded report, scientific authority, open-ended research

**Explanation member**:
One request's part of an Explanatory synthesis, written from that request's own rows: for each row a
claim word, which is the row's recorded state in the researcher's words and is filled by code, and a
short why that says why the request ended so, with limitations over the request's Qualifications. One
conclusion covers every member and the Run's own Qualifications. A member that its decision stopped
is "not explained".
_Avoid_: finding, sub-report, Investigation member

**Readable report**:
The researcher's one-page Run report of one Grounded report and its latest Explanatory synthesis:
retrieved facts copied by code from the report first, then the synthesis's checked words in their
own marked section, identifiers among the references. It is stored nowhere and refuses a row that
disagrees with its report.
_Avoid_: summary report, explanation, Grounded report

**Artifact admission**:
Fresh validation that immutable researcher-supplied or retained donor bytes have the identity,
roles, compatibility, and placement required for the current Run. Successful admission may satisfy
a preparation milestone without recomputing the artifact; it never reuses prior Execution
authority.
_Avoid_: Run resume, trusted path, old receipt reuse

**Reference source identity** (0035):
What Execution approval binds about the reference before any byte is acquired: the assembly
accession, the asset, the publisher's exact file URL, the publisher's md5 for that file and its
size. The NCBI adapter observes these values; nobody types them. A supplied prepared index matches
the approved reference by accession and source md5.
_Avoid_: expected sha256, accepted plain FASTA, typed reference identity

**Acquired identity** (0035, 0029):
The reference bytes as measured when they were acquired: the compressed file's SHA-256 (derived
while its md5 was checked) and the plain FASTA's size and SHA-256. It is recorded as a receipt of
approved work and never mutates the approval. Input storage records it when the controller streamed
the source (a supplied local FASTA or index); otherwise the CPU preparation job acquires the source
from the approved URL and `ReferencePrepared` records it.
_Avoid_: approved checksum, pre-approval identity

**CPU preparation review** (0031):
The execution operation's scoped Review junction for a missing, drifted or inadmissible prepared
reference when CPU preparation lacks authority. It presents that gap and asks for the declared
preparation, amendment or abort decision. It neither resolves another failure nor turns an amended
value into verified evidence; unanswered, only affected preparation waits.
_Avoid_: failure-review approval, automatic paid fallback, global Run rejection


**Artifact verification proof**:
Evidence binding an artifact's established content identity to its creation or admission assurance
and controlled origin, transfer and ownership lineage. Trusted reuse retains that origin and is
separate from the current Run's scientific and execution authority.
_Avoid_: Trusted pathname, approval, scientific admission

**Verification checkpoint**:
A designated artifact boundary that establishes content identity at creation or admission,
or performs explicitly selected independent verification with a recorded reason.
_Avoid_: Slice checkpoint, offline test checkpoint

**Object verification mode**:
The assurance choice for a local or remote artifact: trust its established producer proof
for ordinary use, or explicitly select independent verification with a recorded reason. The choice
describes integrity assurance, not scientific eligibility or permission to act.
_Avoid_: Trust level, verification strength

**Transfer proof**:
Evidence of an actual transfer, binding its source artifact and proof to the destination and
the transport assurance under which it completed. It distinguishes retained source identity
from any new verification actually performed.
_Avoid_: Checksum, ETag

**Run object storage**:
The selected Nebius Object Storage location that holds the current Run's staged inputs,
prepared-reference artifacts, worker receipts, and execution outputs. S3 compatibility describes
Adapter mechanics only and does not name the product responsibility.
_Avoid_: S3 workflow, Amazon bucket

## Lineage and reuse

**Donor**:
Existing code, tests, documents, or retained runs inspected for bounded evidence and selective
adaptation. A Donor never defines current authority.
_Avoid_: Template, implementation contract

**Fork** (0031):
A new Run and Journey with exact causal lineage from a retained parent prefix and fresh child
decision authority. It starts idle and can continue the same inquiry or a different one from a
supported checked boundary (a closed Pass). Inherited evidence keeps its original grounding; the parent
is unchanged. "Child" names only the Run a Fork creates. A Fork is never an in-Run retry: waiting at a
Review, retries and failure-module corrections continue the same Run.
_Avoid_: crash resume, mutation of the parent Run, development clone, restart, causal child,
retained-execution continuation, causal Finalization continuation

**Reconstruction** (0031, 0029):
Seeding a Fork from retained artifact identities, locations and established creation proof named
by an earlier Journey, preserving causal lineage without replaying or mutating the parent. It grants
no new child authority and is distinct from adopting a running job.
_Avoid_: parent replay, failed-Run resume, implicit recovery

**Run provenance**:
The typed lineage classification that distinguishes a fresh causal Run, a Run a Fork created, and
a non-fresh development clone without inferring one from filesystem placement.
`unestablished` records that the available Journey evidence does not establish a provenance
classification; it is not another kind of Run and makes no freshness or development-clone claim.
_Avoid_: Freshness flag, directory label

## Documents

**Governing idea** (0028):
One repo-wide rule with its problem, solution, stories, decisions, instances, example, tests and
scope; the content an ADR file holds. Selected by matching a ticket's paths against `applies_to`.
_Avoid_: Spec table, clause list, amendment note, idea page

**ADR** (0028):
The file under `docs/adr/` that holds one Governing idea in the ten-section shape; the read path
holds only current truth and ids are never reused.
_Avoid_: Spec, changelog, design reference, chain log

**Design reference** (0028):
The current-contract page of one surface under `docs/design/<surface>.md`, pointing at the
governing ideas it instantiates; may be a pointer page while the surface settles.
_Avoid_: Implementation note, ticket receipt, ADR, stale bridge
