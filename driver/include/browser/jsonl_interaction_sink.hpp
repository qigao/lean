#pragma once

#include "browser/interaction_journal.hpp"

#include <ostream>

namespace browser::interaction {

/** JSONL serializer for the Lean Driver-facing interaction journal contract. */
class JsonlInteractionSink final : public InteractionSink {
 public:
  explicit JsonlInteractionSink(std::ostream& output);

  void write(const InteractionRecord& record) override;

 private:
  static const char* event_name(InteractionKind kind) noexcept;

  std::ostream* output_;
};

}  // namespace browser::interaction
