#include "browser/jsonl_interaction_sink.hpp"

#include <cstdint>
#include <optional>

namespace browser::interaction {
namespace {

template <typename T>
void write_optional_number(std::ostream& out, const char* name, const std::optional<T>& value) {
  if (value) {
    out << ",\"" << name << "\":" << *value;
  }
}

}  // namespace

JsonlInteractionSink::JsonlInteractionSink(std::ostream& output) : output_(&output) {
  *output_ << "{\"kind\":\"interaction-journal\",\"version\":1}\n";
  output_->flush();
}

const char* JsonlInteractionSink::event_name(InteractionKind kind) noexcept {
  switch (kind) {
    case InteractionKind::action_started:
      return "actionStarted";
    case InteractionKind::action_finished:
      return "actionFinished";
    case InteractionKind::cdp_request:
      return "cdpRequest";
    case InteractionKind::cdp_response:
      return "cdpResponse";
    case InteractionKind::deadline_armed:
      return "deadlineArmed";
    case InteractionKind::timer_expired:
      return "timerExpired";
    case InteractionKind::input_dispatch:
      return "inputDispatch";
    case InteractionKind::human_input:
      return "humanInput";
    case InteractionKind::policy_issued:
      return "policyIssued";
    case InteractionKind::policy_delivered:
      return "policyDelivered";
    case InteractionKind::epoch_recreated:
      return "epochRecreated";
    case InteractionKind::actor_destroyed:
      return "actorDestroyed";
  }
  return "unknown";
}

void JsonlInteractionSink::write(const InteractionRecord& record) {
  auto& out = *output_;
  out << "{\"kind\":\"interaction\""
      << ",\"seq\":" << record.seq
      << ",\"lane\":" << record.lane
      << ",\"event\":\"" << event_name(record.kind) << "\"";

  write_optional_number(out, "action", record.action);
  write_optional_number(out, "call", record.call);
  write_optional_number(out, "expiresAt", record.expires_at);
  write_optional_number(out, "now", record.now);
  write_optional_number(out, "eventId", record.event_id);
  write_optional_number(out, "correlationId", record.correlation_id);
  write_optional_number(out, "depth", record.depth);
  write_optional_number(out, "page", record.page);

  out << "}\n";
  out.flush();
}

}  // namespace browser::interaction
