import Browser.Interaction.JournalConformance

namespace Browser.Interaction

private def failed {α : Type} : Except String α → Bool
  | .error _ => true
  | .ok _ => false

private def interleavedValid : String :=
  "{\"kind\":\"interaction-journal\",\"version\":1}\n" ++
  "{\"kind\":\"interaction\",\"seq\":1,\"lane\":1,\"event\":\"actionStarted\",\"action\":2}\n" ++
  "{\"kind\":\"interaction\",\"seq\":2,\"lane\":2,\"event\":\"actionStarted\",\"action\":7}\n" ++
  "{\"kind\":\"interaction\",\"seq\":3,\"lane\":1,\"event\":\"deadlineArmed\",\"action\":2,\"expiresAt\":10}\n" ++
  "{\"kind\":\"interaction\",\"seq\":4,\"lane\":1,\"event\":\"timerExpired\",\"action\":2,\"now\":10}\n" ++
  "{\"kind\":\"interaction\",\"seq\":5,\"lane\":2,\"event\":\"inputDispatch\",\"action\":7}\n"

private def outOfOrder : String :=
  "{\"kind\":\"interaction-journal\",\"version\":1}\n" ++
  "{\"kind\":\"interaction\",\"seq\":2,\"lane\":1,\"event\":\"actionStarted\",\"action\":2}\n" ++
  "{\"kind\":\"interaction\",\"seq\":1,\"lane\":1,\"event\":\"humanInput\"}\n"

private def laneLocalViolation : String :=
  "{\"kind\":\"interaction-journal\",\"version\":1}\n" ++
  "{\"kind\":\"interaction\",\"seq\":1,\"lane\":1,\"event\":\"actionStarted\",\"action\":2}\n" ++
  "{\"kind\":\"interaction\",\"seq\":2,\"lane\":2,\"event\":\"actionStarted\",\"action\":7}\n" ++
  "{\"kind\":\"interaction\",\"seq\":3,\"lane\":1,\"event\":\"humanInput\"}\n" ++
  "{\"kind\":\"interaction\",\"seq\":4,\"lane\":2,\"event\":\"inputDispatch\",\"action\":7}\n" ++
  "{\"kind\":\"interaction\",\"seq\":5,\"lane\":1,\"event\":\"inputDispatch\",\"action\":2}\n"

example : (verifyInteractionJournalJsonl interleavedValid).isOk = true := by
  native_decide

example : failed (verifyInteractionJournalJsonl outOfOrder) = true := by
  native_decide

example : failed (verifyInteractionJournalJsonl laneLocalViolation) = true := by
  native_decide

end Browser.Interaction
