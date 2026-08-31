# Situated Long-Term Memory V12 Design

## Objective

Persist each situated agent's observed history as durable, structured, private
long-term memory. Early observations must remain available after the in-memory
story has advanced and after the process has restarted. SQLite ordinary tables
are the source of truth; an FTS5 index provides rebuildable lexical retrieval.

V12 is a storage and retrieval boundary. It does not inject recalled memories
into the V11 belief update or POMDP policy. Agent-private retrieval influencing
decisions is V13.

## Privacy and provenance boundary

The only ingestion input is `perspective_timeline(story, agent_id)`. The objective
timeline is never copied wholesale, and one agent's rows are never inferred from
another agent's observations. Every public read, search, activation, and ingestion
operation requires an explicit `agent_id`.

Each memory preserves the complete observed event and observation provenance:

- deterministic memory and observation IDs;
- story model ID and exact model hash;
- event ID and exact event hash;
- round, sequence, action, actor, place, target, success, and outcome;
- structured evidence details and causal event IDs;
- the observation channel;
- declared confidence and salience from the exact memory policy hash;
- a deterministic summary used for lexical indexing.

The SQLite file is a trusted simulator artifact, not an encryption boundary. Code
with direct database access can inspect all rows. API-level scoping prevents normal
simulation code from accidentally retrieving another agent's memory.

## Memory policy

`SituatedMemoryPolicy` declares one `MemoryChannelPolicy` for every V10 observation
channel. A channel policy supplies confidence and salience in the closed interval
`[0, 1]`. The standard policy is explicit and hashable:

| Channel | Confidence | Salience |
| --- | ---: | ---: |
| self | 1.00 | 0.50 |
| visual | 0.90 | 0.65 |
| auditory | 0.70 | 0.75 |
| inspection | 1.00 | 1.00 |

These are modeling defaults, not learned psychological facts. Callers can provide
a different complete policy, and each stored row records which exact policy made
the attribution.

## Persistence model

Schema version 1 contains:

- `memory_metadata`, carrying the schema version;
- `memory_records`, the authoritative structured records;
- `memory_fts`, an external-content FTS5 table with the `trigram` tokenizer;
- insert, update, and delete triggers keeping indexed text synchronized.

Writes are transactional and idempotent on `(agent_id, observation_id)`. Replaying
the same observation with the same record hash is a no-op. Reusing that identity
with different content is rejected as a deterministic-history conflict.

Memory activation is soft deletion. An inactive row remains auditable in the
ordinary table but is excluded by default from list and search operations. It can
be reactivated only through an operation scoped to its owning agent.

`rebuild_situated_memory_index` reconstructs FTS5 exclusively from authoritative
rows. Database initialization fails clearly when the Python SQLite build lacks
FTS5 or the trigram tokenizer.

## Retrieval contract

`SituatedMemoryQuery` always names exactly one agent and supports:

- optional plain-text substring search;
- actor, place, event kind, and observation channel filters;
- round interval and minimum confidence filters;
- active-only retrieval by default;
- a validated result limit.

Text is treated as literal user text, not raw FTS query syntax. Queries of at least
three characters use trigram FTS5 for candidate matching, then receive a
lower-is-better agent-local literal-occurrence score computed from the authoritative
summary. The score and ordering never depend on another agent's FTS corpus. One- or
two-character queries fall back to a parameterized `LIKE` scan because trigram
indexes cannot match them. Stable tie-breaking is salience descending, round
descending, then memory ID. Structured-only queries do not touch FTS5. NUL is
rejected before it reaches the FTS query grammar.

## Determinism

Canonical JSON uses sorted keys, compact separators, and UTF-8-preserving text.
Records are derived only from immutable V10 perspective events and the declared
policy. Re-ingesting the same story and policy produces the same records and
hashes, regardless of process or database path.

## Office acceptance scenario

The existing office story is persisted independently for Alice, Bob, and Dana:

1. Alice can retrieve her private inspection of the restructuring memo.
2. Bob cannot retrieve Alice's inspection, but can retrieve Alice's later telling.
3. Dana, isolated in the manager office, retrieves neither event.
4. Ingesting the story twice inserts no duplicate rows.
5. Closing and reopening the database preserves the same record hashes.
6. Alice's round-one inspection remains retrievable after many later rounds.
7. Soft-deactivated memories disappear from normal search and can be reactivated.
8. Rebuilding FTS5 preserves the same scoped search results.

## Deferred work

- V13 retrieval injection into each agent's private belief and planning context;
- semantic embeddings or external vector databases;
- consolidation, contradiction resolution, relationship learning, and forgetting
  policies beyond explicit activation (V14);
- encryption or operating-system isolation between agent stores;
- LSTM, Transformer, or learned sequence-state models.
