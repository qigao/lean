import Browser.Interaction.Feedback

namespace Browser.Interaction

example : decideFeedback .success = .complete := by rfl
example : decideFeedback .transient = .wait := by rfl
example : decideFeedback .stale = .recover .reResolve := by rfl
example : decideFeedback .conflict = .suspend := by rfl
example : decideFeedback .unavailable = .recover .reattachSession := by rfl
example : decideFeedback .timeout = .failSafe := by rfl
example : decideFeedback .terminal = .recover .recreatePage := by rfl
example : decideFeedback .unknown = .failSafe := by rfl

theorem feedback_decision_total (feedback : FeedbackClass) :
    decideFeedback feedback = .complete ∨
    decideFeedback feedback = .wait ∨
    decideFeedback feedback = .retry ∨
    (∃ recovery, decideFeedback feedback = .recover recovery) ∨
    decideFeedback feedback = .suspend ∨
    decideFeedback feedback = .failSafe := by
  cases feedback <;> simp [decideFeedback]

end Browser.Interaction
