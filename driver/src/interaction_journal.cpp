#include "browser/interaction_journal.hpp"

#include <utility>

namespace browser::interaction {

InteractionJournal::InteractionJournal(InteractionSink& sink) noexcept : sink_(&sink) {}

void InteractionJournal::emit(InteractionRecord record) {
  std::lock_guard<std::mutex> lock(mutex_);
  record.seq = next_seq_;
  sink_->write(record);
  ++next_seq_;
}

void InteractionJournal::action_started(LaneId lane, ActionId action) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::action_started;
  record.action = action;
  emit(std::move(record));
}

void InteractionJournal::action_finished(LaneId lane, ActionId action) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::action_finished;
  record.action = action;
  emit(std::move(record));
}

void InteractionJournal::cdp_request(LaneId lane, ActionId action, CdpCallId call) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::cdp_request;
  record.action = action;
  record.call = call;
  emit(std::move(record));
}

void InteractionJournal::cdp_response(LaneId lane, ActionId action, CdpCallId call) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::cdp_response;
  record.action = action;
  record.call = call;
  emit(std::move(record));
}

void InteractionJournal::deadline_armed(LaneId lane, ActionId action, Time expires_at) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::deadline_armed;
  record.action = action;
  record.expires_at = expires_at;
  emit(std::move(record));
}

void InteractionJournal::timer_expired(LaneId lane, ActionId action, Time now) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::timer_expired;
  record.action = action;
  record.now = now;
  emit(std::move(record));
}

void InteractionJournal::input_dispatch(LaneId lane, ActionId action) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::input_dispatch;
  record.action = action;
  emit(std::move(record));
}

void InteractionJournal::human_input(LaneId lane) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::human_input;
  emit(std::move(record));
}

void InteractionJournal::policy_issued(
    LaneId lane,
    EventId event_id,
    CorrelationId correlation_id,
    std::uint32_t depth,
    PageId page) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::policy_issued;
  record.event_id = event_id;
  record.correlation_id = correlation_id;
  record.depth = depth;
  record.page = page;
  emit(std::move(record));
}

void InteractionJournal::policy_delivered(LaneId lane) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::policy_delivered;
  emit(std::move(record));
}

void InteractionJournal::epoch_recreated(LaneId lane) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::epoch_recreated;
  emit(std::move(record));
}

void InteractionJournal::actor_destroyed(LaneId lane) {
  InteractionRecord record;
  record.lane = lane;
  record.kind = InteractionKind::actor_destroyed;
  emit(std::move(record));
}

}  // namespace browser::interaction
