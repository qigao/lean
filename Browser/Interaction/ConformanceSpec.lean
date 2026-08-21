import Browser.Interaction.Conformance

namespace Browser.Interaction

private def validTrace : String :=
  "{\"kind\":\"interaction\",\"event\":\"actionStarted\",\"action\":2}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"cdpRequest\",\"action\":2,\"call\":17}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"cdpResponse\",\"action\":1,\"call\":17}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"deadlineArmed\",\"action\":2,\"expiresAt\":100}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"timerExpired\",\"action\":1,\"now\":999}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"inputDispatch\",\"action\":2}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"humanInput\"}\n"

private def terminalViolation : String :=
  "{\"kind\":\"interaction\",\"event\":\"actionStarted\",\"action\":2}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"actorDestroyed\"}\n" ++
  "{\"kind\":\"interaction\",\"event\":\"inputDispatch\",\"action\":2}\n"

private def forgedPolicyDelivery : String :=
  "{\"kind\":\"interaction\",\"event\":\"policyDelivered\"}\n"

example :
    (parseInteractionLine
      "{\"kind\":\"interaction\",\"event\":\"cdpResponse\",\"action\":2,\"call\":17}").isOk = true := by
  native_decide

example : (verifyInteractionJsonl validTrace).isOk = true := by
  native_decide

example : (verifyInteractionJsonl terminalViolation).isError = true := by
  native_decide

example : (verifyInteractionJsonl forgedPolicyDelivery).isError = true := by
  native_decide

end Browser.Interaction
