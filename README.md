# Lean Browser Runtime Safety Model

This repository formalizes safety properties for a stateful asynchronous CDP browser Driver in Lean 4.

The primary proof surface is **interaction + decoder/feedback/recovery contracts**, not a second full implementation of the Browser Driver. The larger runtime model remains as integration/reference evidence.
