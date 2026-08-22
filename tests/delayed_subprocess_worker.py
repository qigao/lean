from __future__ import annotations

import time

from narrative_dynamics.subprocess_worker import main


if __name__ == "__main__":
    time.sleep(2.0)
    raise SystemExit(main())
