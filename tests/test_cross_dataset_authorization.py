from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.cross_dataset_authorization import (
    TransferAuthorizationReceipt,
    parse_transfer_authorization,
)
from tests.cross_dataset_transfer_fixtures import (
    authorization_comment,
    dual_preflight,
    final_started_event,
)


class CrossDatasetAuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _parse(self, comment=None, **overrides: object):
        preflight, store = dual_preflight(self.root / f"case-{len(tuple(self.root.iterdir()))}")
        values: dict[str, object] = {
            "issue_number": 43,
            "comment": (
                authorization_comment(preflight)
                if comment is None
                else comment(preflight)
            ),
            "authorized_owner": "qigao",
            "preflight": preflight,
            "lock_commit": "b" * 40,
            "store": store,
        }
        values.update(overrides)
        return parse_transfer_authorization(**values)

    def test_exact_post_preflight_owner_comment_builds_receipt(self) -> None:
        preflight, store = dual_preflight(self.root / "valid")
        receipt = parse_transfer_authorization(
            issue_number=43,
            comment=authorization_comment(preflight),
            authorized_owner="qigao",
            preflight=preflight,
            lock_commit="b" * 40,
            store=store,
        )
        self.assertEqual(receipt.issue_number, 43)
        self.assertEqual(receipt.author_login, "qigao")
        self.assertEqual(
            receipt.pre_authorization_ledger_head,
            preflight.ledger_head_hash,
        )

    def test_edited_predated_or_identity_drifted_comment_is_rejected(self) -> None:
        cases = (
            lambda preflight: authorization_comment(
                preflight,
                updated_at="2026-08-30T14:01:00Z",
            ),
            lambda preflight: authorization_comment(
                preflight,
                created_at="2026-08-30T13:00:00Z",
                updated_at="2026-08-30T13:00:00Z",
            ),
            lambda preflight: authorization_comment(
                preflight,
                body=authorization_comment(preflight)["body"].replace(
                    preflight.scientific_revision,
                    "c" * 40,
                ),
            ),
        )
        for index, comment in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self._parse(comment)

    def test_issue_author_deleted_and_url_identity_are_exact(self) -> None:
        with self.assertRaisesRegex(ValueError, "issue 43"):
            self._parse(issue_number=44)
        for field, value in (
            ("author_login", "someone-else"),
            ("deleted", True),
            ("html_url", "https://github.com/qigao/lean/issues/43#issuecomment-1"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self._parse(
                        lambda preflight, field=field, value=value: authorization_comment(
                            preflight,
                            **{field: value},
                        )
                    )

    def test_body_rejects_surrounding_reordered_missing_duplicate_or_crlf_text(self) -> None:
        def mutate(preflight, variant):
            body = str(authorization_comment(preflight)["body"])
            lines = body.split("\n")
            mutations = {
                "surrounding": f"prefix\n{body}",
                "reordered": "\n".join((lines[0], lines[2], lines[1], *lines[3:])),
                "missing": "\n".join(lines[:-1]),
                "duplicate": "\n".join(lines + [lines[-1]]),
                "crlf": body.replace("\n", "\r\n"),
            }
            return authorization_comment(preflight, body=mutations[variant])

        for variant in ("surrounding", "reordered", "missing", "duplicate", "crlf"):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "exact six-line"):
                    self._parse(lambda preflight, variant=variant: mutate(preflight, variant))

    def test_each_bound_identity_must_match_exactly(self) -> None:
        def drift(preflight, line_name, value):
            lines = str(authorization_comment(preflight)["body"]).split("\n")
            return authorization_comment(
                preflight,
                body="\n".join(
                    value if line.startswith(f"{line_name}=") else line
                    for line in lines
                ),
            )

        cases = (
            ("scientific_sha", "scientific_sha=" + "c" * 40),
            ("lock_commit", "lock_commit=" + "d" * 40),
            ("preflight_hash", "preflight_hash=sha256:" + "e" * 64),
            ("brier_release_hash", "brier_release_hash=sha256:" + "e" * 64),
            ("log_release_hash", "log_release_hash=sha256:" + "e" * 64),
            ("lock_commit", "lock_commit=ABC"),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    self._parse(
                        lambda preflight, name=name, value=value: drift(
                            preflight,
                            name,
                            value,
                        )
                    )

    def test_timestamp_must_be_strictly_after_preflight(self) -> None:
        for timestamp in ("2026-08-30T12:59:59Z", "2026-08-30T13:00:00Z"):
            with self.subTest(timestamp=timestamp):
                with self.assertRaisesRegex(ValueError, "strictly after"):
                    self._parse(
                        lambda preflight, timestamp=timestamp: authorization_comment(
                            preflight,
                            created_at=timestamp,
                            updated_at=timestamp,
                        )
                    )

    def test_stale_head_and_receipt_reuse_after_started_are_rejected(self) -> None:
        preflight, store = dual_preflight(self.root / "consumed")
        head = store.compare_and_append(
            store.head(),
            final_started_event(parent=store.head()),
        )
        self.assertNotEqual(head, preflight.ledger_head_hash)
        with self.assertRaisesRegex(ValueError, "stale ledger"):
            parse_transfer_authorization(
                issue_number=43,
                comment=authorization_comment(preflight),
                authorized_owner="qigao",
                preflight=preflight,
                lock_commit="b" * 40,
                store=store,
            )

    def test_receipt_payload_round_trip_is_strict(self) -> None:
        receipt = self._parse()
        payload = receipt.to_payload()
        self.assertEqual(TransferAuthorizationReceipt.from_payload(payload), receipt)
        with self.assertRaises(ValueError):
            TransferAuthorizationReceipt.from_payload({**payload, "unknown": True})


if __name__ == "__main__":
    unittest.main()
