import Browser.Interaction.JournalConformance

open Browser.Interaction

private def usage : String :=
  "usage: browser-driver-interaction-journal-check <journal.jsonl>"

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
      let text ← IO.FS.readFile path
      match verifyInteractionJournalJsonl text with
      | .ok _ =>
          IO.println s!"browser driver interaction journal: accepted {path}"
          return 0
      | .error message =>
          IO.eprintln s!"browser driver interaction journal rejected: {message}"
          return 1
  | _ =>
      IO.eprintln usage
      return 2
