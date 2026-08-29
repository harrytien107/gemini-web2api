import http.client
import base64
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from urllib.parse import parse_qs

from gemini_web2api.config import CONFIG, DEFAULT_CONFIG
from gemini_web2api.gemini import (
    _build_headers,
    _build_payload,
    _get_url,
    GeminiUpstreamError,
    auth_status,
    conversation_info,
    create_conversation,
    extract_response_text,
    generate,
    generate_stream,
    refresh_auth,
    reset_conversation,
)
from gemini_web2api import gemini as gemini_module
from gemini_web2api import multimodal
from gemini_web2api.models import resolve_model_chain
from gemini_web2api.server import (
    GeminiHandler,
    ThreadedServer,
    _generate_attempts,
    _generate_stream_attempts,
    _commit_chat_messages,
    _incremental_chat_messages,
    _model_catalog,
)
from gemini_web2api.tools import google_contents_to_prompt, messages_to_prompt


def _decode_payload(payload):
    outer = json.loads(parse_qs(payload)["f.req"][0])
    return json.loads(outer[1])


def _decode_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event_type = next(
            (line[len("event: "):] for line in lines if line.startswith("event: ")),
            None,
        )
        data = next(
            (line[len("data: "):] for line in lines if line.startswith("data: ")),
            None,
        )
        if event_type and data:
            events.append((event_type, json.loads(data)))
    return events


class AuthReloadTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.auth_path = os.path.join(self.temp_dir.name, "gemini-auth.json")
        CONFIG.update({
            "cookie_file": self.auth_path,
            "auth_user": "config-user",
            "xsrf_token": "config-xsrf",
            "gemini_bl": "config-bl",
            "log_requests": False,
        })

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)
        refresh_auth(force=True)
        self.temp_dir.cleanup()

    def write_auth(self, **overrides):
        payload = {
            "cookie": "SID=fake; SAPISID=fake-sapisid",
            "sapisid": "fake-sapisid",
            "auth_user": None,
            "xsrf_token": "auth-xsrf",
            "gemini_bl": "auth-bl",
            **overrides,
        }
        with open(self.auth_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.utime(self.auth_path, None)
        return payload

    def test_extension_schema_overrides_auth_config_fields(self):
        self.write_auth()
        auth = refresh_auth(force=True)

        self.assertEqual(auth["cookie"], "SID=fake; SAPISID=fake-sapisid")
        self.assertEqual(auth["sapisid"], "fake-sapisid")
        self.assertIsNone(auth["auth_user"])
        self.assertEqual(auth["xsrf_token"], "auth-xsrf")
        self.assertEqual(auth["gemini_bl"], "auth-bl")
        self.assertNotIn("/u/", _get_url(auth))
        self.assertIn("bl=auth-bl", _get_url(auth))
        self.assertNotIn("X-Goog-AuthUser", _build_headers(auth))
        self.assertEqual(parse_qs(_build_payload("hello", 1, 4, auth=auth))["at"][0], "auth-xsrf")

    def test_missing_metadata_falls_back_to_config(self):
        self.write_auth(auth_user="2", xsrf_token=None)
        with open(self.auth_path, "w", encoding="utf-8") as f:
            json.dump({"cookie": "SAPISID=fake-sapisid", "sapisid": "fake-sapisid"}, f)

        auth = refresh_auth(force=True)

        self.assertEqual(auth["auth_user"], "config-user")
        self.assertEqual(auth["xsrf_token"], "config-xsrf")
        self.assertEqual(auth["gemini_bl"], "config-bl")

    def test_empty_generated_template_uses_anonymous_auth(self):
        with open(self.auth_path, "w", encoding="utf-8") as f:
            json.dump({
                "cookie": "",
                "sapisid": None,
                "auth_user": None,
                "xsrf_token": None,
                "gemini_bl": None,
            }, f)

        auth = refresh_auth(force=True)
        status = auth_status()

        self.assertEqual(auth["cookie"], "")
        self.assertIsNone(auth["sapisid"])
        self.assertTrue(status["exists"])
        self.assertFalse(status["loaded"])
        self.assertIsNone(status["error"])

    def test_hot_reload_and_invalid_json_keep_last_known_good(self):
        self.write_auth(xsrf_token="first")
        self.assertEqual(refresh_auth(force=True)["xsrf_token"], "first")
        self.write_auth(xsrf_token="second", cookie="SID=second; SAPISID=second")
        self.assertEqual(refresh_auth()["xsrf_token"], "second")

        with open(self.auth_path, "w", encoding="utf-8") as f:
            f.write("{")
        os.utime(self.auth_path, None)
        auth = refresh_auth()

        self.assertEqual(auth["xsrf_token"], "second")
        self.assertEqual(auth["cookie"], "SID=second; SAPISID=second")
        self.assertIsNotNone(auth_status()["error"])

    def test_concurrent_refresh_returns_complete_snapshots(self):
        self.write_auth(auth_user="3", xsrf_token="thread-xsrf", gemini_bl="thread-bl")
        refresh_auth(force=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            snapshots = list(pool.map(lambda _: refresh_auth(), range(64)))

        self.assertTrue(all(snapshot["cookie"] == "SID=fake; SAPISID=fake-sapisid" for snapshot in snapshots))
        self.assertTrue(all(snapshot["auth_user"] == "3" for snapshot in snapshots))
        self.assertTrue(all(snapshot["xsrf_token"] == "thread-xsrf" for snapshot in snapshots))
        self.assertTrue(all(snapshot["gemini_bl"] == "thread-bl" for snapshot in snapshots))

    def test_extract_response_text_accepts_short_wrb_line(self):
        inner = [None] * 5
        inner[4] = [[None, ["API_TEST_OK"]]]
        raw = json.dumps([["wrb.fr", None, json.dumps(inner)]])

        self.assertLess(len(raw), 200)
        self.assertEqual(extract_response_text(raw), "API_TEST_OK")

    def test_extract_response_text_reports_nested_upstream_error(self):
        frame = ["wrb.fr", None, None, None, None, [3, None, [[None, [1003]]]]]

        with self.assertRaisesRegex(RuntimeError, "error 1003"):
            extract_response_text(json.dumps([frame]))

    @mock.patch("gemini_web2api.gemini.urllib.request.urlopen")
    def test_generate_retries_empty_upstream_response(self, urlopen):
        CONFIG["retry_attempts"] = 2
        CONFIG["retry_delay_sec"] = 0
        empty_response = mock.Mock()
        empty_response.read.return_value = b""
        good_response = mock.Mock()
        inner = [None] * 5
        inner[4] = [[None, ["API_TEST_OK"]]]
        good_response.read.return_value = json.dumps(
            [["wrb.fr", None, json.dumps(inner)]]
        ).encode()
        urlopen.side_effect = [empty_response, good_response]

        self.assertEqual(generate("test", 1, 4), "API_TEST_OK")
        self.assertEqual(urlopen.call_count, 2)

    @mock.patch("gemini_web2api.gemini.log")
    @mock.patch("gemini_web2api.gemini.urllib.request.urlopen")
    def test_generate_does_not_retry_terminal_upstream_rejection(self, urlopen, log):
        CONFIG["retry_attempts"] = 3
        rejected = mock.Mock()
        rejected.read.return_value = b"BardErrorInfo [1100]"
        urlopen.return_value = rejected

        with self.assertRaisesRegex(GeminiUpstreamError, "error 1100"):
            generate("secret prompt", 1, 4)

        urlopen.assert_called_once()
        diagnostic = log.call_args.args[0]
        self.assertIn("code=1100", diagnostic)
        self.assertIn("prompt_chars=13", diagnostic)
        self.assertIn("upstream_thread=no", diagnostic)
        self.assertNotIn("secret prompt", diagnostic)


    @mock.patch("gemini_web2api.gemini.HAS_HTTPX", True)
    @mock.patch("gemini_web2api.gemini.log")
    @mock.patch("gemini_web2api.gemini._get_httpx_client")
    def test_generate_stream_logs_redacted_rejection_diagnostics(self, get_client, log):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_text.return_value = iter(["BardErrorInfo [1096]"])
        get_client.return_value.stream.return_value = response

        with self.assertRaisesRegex(GeminiUpstreamError, "error 1096"):
            list(generate_stream("private stream prompt", 1, 4, ["image-ref"]))

        diagnostic = log.call_args.args[0]
        self.assertIn("code=1096", diagnostic)
        self.assertIn("prompt_chars=21", diagnostic)
        self.assertIn("attachments=1", diagnostic)
        self.assertNotIn("private stream prompt", diagnostic)
        self.assertNotIn("image-ref", diagnostic)


class MultimodalAuthTests(unittest.TestCase):
    def setUp(self):
        self.original_cache = dict(multimodal._page_tokens_cache)
        multimodal._page_tokens_cache.update({"tokens": {}, "ts": 0, "auth_key": None})

    def tearDown(self):
        multimodal._page_tokens_cache.clear()
        multimodal._page_tokens_cache.update(self.original_cache)

    @mock.patch("gemini_web2api.multimodal._upload_opener.open")
    def test_page_tokens_use_account_scoped_url_and_headers(self, open_page):
        response = mock.Mock()
        response.read.return_value = (
            b'<script>"qKIAYe":"push-token","Ylro7b":"pctx-token",'
            b'"SNlM0e":"fresh-at","cfb2h":"fresh-bl",'
            b'"FdrFJe":"fresh-session"</script>'
        )
        open_page.return_value = response
        auth = {
            "cookie": "SID=fake; SAPISID=fake-sapisid",
            "sapisid": "fake-sapisid",
            "auth_user": "3",
        }

        tokens = multimodal._get_page_tokens(auth)

        self.assertEqual(tokens, {
            "push_id": "push-token",
            "pctx": "pctx-token",
            "at": "fresh-at",
            "bl": "fresh-bl",
            "session_id": "fresh-session",
        })
        request = open_page.call_args.args[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(request.full_url, "https://gemini.google.com/u/3/app")
        self.assertEqual(headers["x-goog-authuser"], "3")
        self.assertEqual(headers["referer"], "https://gemini.google.com/u/3/app")
        self.assertEqual(headers["origin"], "https://gemini.google.com")
        self.assertEqual(headers["cookie"], auth["cookie"])
        self.assertTrue(headers["authorization"].startswith("SAPISIDHASH "))

    @mock.patch("gemini_web2api.multimodal._get_page_tokens")
    def test_page_token_cache_invalidates_when_auth_changes(self, get_page_tokens):
        get_page_tokens.side_effect = [
            {"push_id": "first", "pctx": "first"},
            {"push_id": "second", "pctx": "second"},
        ]
        first = {"cookie": "SID=first", "sapisid": None, "auth_user": "1"}
        second = {"cookie": "SID=second", "sapisid": None, "auth_user": "1"}

        self.assertEqual(multimodal._cached_page_tokens(first)["push_id"], "first")
        self.assertEqual(multimodal._cached_page_tokens(first)["push_id"], "first")
        self.assertEqual(multimodal._cached_page_tokens(second)["push_id"], "second")
        self.assertEqual(get_page_tokens.call_count, 2)

    @mock.patch("gemini_web2api.multimodal._cached_page_tokens")
    @mock.patch("gemini_web2api.multimodal.refresh_auth")
    def test_upload_rejects_missing_page_metadata(self, refresh_auth_mock, cached_tokens):
        refresh_auth_mock.return_value = {
            "cookie": "SID=fake",
            "sapisid": None,
            "auth_user": "2",
        }
        cached_tokens.return_value = {}

        with self.assertRaisesRegex(RuntimeError, "omitted upload metadata"):
            multimodal.upload_image(b"image")

    @mock.patch("gemini_web2api.multimodal._upload_opener.open")
    @mock.patch("gemini_web2api.multimodal._cached_page_tokens")
    @mock.patch("gemini_web2api.multimodal.refresh_auth")
    def test_upload_uses_single_multipart_request(
        self, refresh_auth_mock, cached_tokens, open_upload
    ):
        refresh_auth_mock.return_value = {
            "cookie": "SID=fake",
            "sapisid": None,
            "auth_user": "2",
        }
        cached_tokens.return_value = {"push_id": "push-token"}
        open_upload.return_value.read.return_value = b"/uploaded/image-ref"

        ref = multimodal.upload_image(b"image-bytes", "photo.png", "image/png")

        self.assertEqual(ref, "/uploaded/image-ref")
        self.assertEqual(open_upload.call_count, 1)
        request = open_upload.call_args.args[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(request.full_url, "https://content-push.googleapis.com/upload")
        self.assertEqual(request.get_method(), "POST")
        self.assertIn("multipart/form-data; boundary=", headers["content-type"])
        self.assertEqual(headers["push-id"], "push-token")
        self.assertEqual(headers["x-goog-authuser"], "2")
        self.assertNotIn("x-goog-upload-command", headers)
        self.assertIn(b'name="file"; filename="photo.png"', request.data)
        self.assertIn(b"Content-Type: image/png", request.data)
        self.assertIn(b"image-bytes", request.data)


class FileLoggingTests(unittest.TestCase):
    def test_frozen_log_appends_beside_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.dict(CONFIG, {"log_requests": True}), \
                mock.patch.object(gemini_module.sys, "frozen", True, create=True), \
                mock.patch.object(
                    gemini_module.sys,
                    "executable",
                    os.path.join(temp_dir, "gemini-web2api.exe"),
                ):
            gemini_module.log("Model gemini-3.7-flash: success")

            with open(
                os.path.join(temp_dir, "gemini-web2api.log"),
                encoding="utf-8",
            ) as file:
                content = file.read()

        self.assertIn("Model gemini-3.7-flash: success", content)


class PayloadPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_temporary_chats_default_to_disabled(self):
        self.assertIs(DEFAULT_CONFIG["temporary_chats"], False)

    def test_persistent_chat_payload(self):
        CONFIG["temporary_chats"] = False

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [2])
        self.assertIsNone(inner[45])

    def test_runtime_page_metadata_updates_generation_url_and_token(self):
        auth = {
            "auth_user": "",
            "cookie": "SID=fake",
            "sapisid": None,
            "xsrf_token": "stale-at",
            "gemini_bl": "stale-bl",
        }
        from gemini_web2api.gemini import set_runtime_page_metadata
        set_runtime_page_metadata(auth, {
            "at": "fresh-at",
            "bl": "fresh-bl",
            "session_id": "fresh-session",
        })

        payload = parse_qs(_build_payload("hello", 1, 4, auth=auth))
        url = _get_url(auth)

        self.assertEqual(payload["at"], ["fresh-at"])
        self.assertIn("bl=fresh-bl", url)
        self.assertIn("f.sid=fresh-session", url)

    def test_temporary_chat_payload(self):
        CONFIG["temporary_chats"] = True

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [1])
        self.assertEqual(inner[45], 1)

    def test_payload_reuses_conversation_metadata(self):
        auth = {
            "auth_user": "",
            "cookie": "SID=conversation-test",
            "sapisid": None,
            "xsrf_token": None,
            "gemini_bl": "test-bl",
        }
        reset_conversation()
        gemini_module._update_conversation_metadata(
            json.dumps([["wrb.fr", None, json.dumps([None, ["cid", "rid", "rcid"]])]]),
            auth,
        )

        inner = _decode_payload(_build_payload("follow up", 1, 4, auth=auth))

        self.assertEqual(inner[2][:3], ["cid", "rid", "rcid"])

    def test_payload_includes_uploaded_image_refs(self):
        inner = _decode_payload(_build_payload(
            "describe", 1, 4, [("/uploaded/image-ref", "image.png")],
            request_uuid="REQUEST-UUID",
        ))

        self.assertEqual(inner[0][0], "describe")
        self.assertEqual(inner[0][3], [[["/uploaded/image-ref"], "image.png"]])
        self.assertEqual(len(inner), 81)
        self.assertEqual(inner[4].isalnum(), True)
        self.assertEqual(inner[6], [1])
        self.assertEqual(inner[17], [[0]])
        self.assertEqual(inner[55], [[1]])
        self.assertEqual(inner[59], "REQUEST-UUID")
        self.assertEqual(inner[80], 1)

    def test_generation_headers_include_matching_request_uuid(self):
        headers = _build_headers({
            "auth_user": "",
            "cookie": "",
            "sapisid": "",
        }, "REQUEST-UUID")

        self.assertEqual(
            json.loads(headers["X-Goog-Ext-525005358-Jspb"]),
            ["REQUEST-UUID", 1],
        )


class MessageParsingTests(unittest.TestCase):
    def test_messages_to_prompt_extracts_openai_image_url_data_url(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_messages_to_prompt_extracts_responses_input_image_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe"},
                {"type": "input_image", "image_url": "https://example.com/image.png"},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [("https://example.com/image.png", "image/png")])

    def test_messages_to_prompt_ignores_malformed_image_data_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
            ],
        }])

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])

    def test_google_contents_to_prompt_extracts_inline_image_data(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": image_data}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe\n[Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_google_contents_to_prompt_ignores_malformed_inline_image_data(self):
        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": "%%%"}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])


class ComboModelTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["log_requests"] = False
        CONFIG["model_combos"] = {
            "gemini-combo": ["gemini-3.1-pro", "gemini-3.7-flash"],
            "gemini-deep": [
                "gemini-3.7-flash-thinking@think=0",
                "gemini-3.6-flash-thinking@think=0",
                "gemini-3.5-flash-thinking@think=0",
            ],
            "gemini-balanced-test": {
                "strategy": "round_robin",
                "models": ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"],
            },
        }

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_fallback_combo_is_exposed_when_configured(self):
        self.assertIn("gemini-combo", _model_catalog())
        name, attempts, error = resolve_model_chain("gemini-combo")
        self.assertEqual(name, "gemini-combo")
        self.assertIsNone(error)
        self.assertEqual([attempt[0] for attempt in attempts], ["gemini-3.1-pro", "gemini-3.7-flash"])

    def test_named_combo_is_exposed_and_resolves_think_override(self):
        self.assertIn("gemini-deep", _model_catalog())
        name, attempts, error = resolve_model_chain("gemini-deep")
        self.assertEqual(name, "gemini-deep")
        self.assertIsNone(error)
        self.assertEqual(
            [attempt[0] for attempt in attempts],
            [
                "gemini-3.7-flash-thinking",
                "gemini-3.6-flash-thinking",
                "gemini-3.5-flash-thinking",
            ],
        )
        self.assertTrue(all(attempt[2] == 0 for attempt in attempts))

    def test_round_robin_rotates_first_model_and_keeps_fallbacks(self):
        self.assertIn("gemini-balanced-test", _model_catalog())
        orders = []
        for _ in range(3):
            _, attempts, error = resolve_model_chain("gemini-balanced-test")
            self.assertIsNone(error)
            orders.append([attempt[0] for attempt in attempts])
        self.assertEqual(orders, [
            ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"],
            ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash"],
            ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
        ])

    def test_named_combo_rejects_nested_combo(self):
        CONFIG["model_combos"]["gemini-nested"] = ["gemini-deep"]
        _, attempts, error = resolve_model_chain("gemini-nested")
        self.assertEqual(attempts, [])
        self.assertIn("Invalid combo model", error)

    def test_named_combo_rejects_unknown_strategy(self):
        CONFIG["model_combos"]["gemini-invalid"] = {
            "strategy": "random",
            "models": ["gemini-3.7-flash"],
        }
        _, attempts, error = resolve_model_chain("gemini-invalid")
        self.assertEqual(attempts, [])
        self.assertIn("Invalid combo strategy", error)

    @mock.patch("gemini_web2api.server.log")
    @mock.patch("gemini_web2api.server.generate")
    def test_combo_falls_back_on_error_and_empty_response(self, generate, log):
        generate.side_effect = [RuntimeError("denied"), ""]
        _, attempts, _ = resolve_model_chain("gemini-combo")
        with self.assertRaisesRegex(RuntimeError, "all combo models failed"):
            _generate_attempts("hello", attempts, None, True)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(log.call_args_list, [
            mock.call("Model gemini-3.1-pro: calling"),
            mock.call("Model gemini-3.1-pro: failed (denied)"),
            mock.call("Model gemini-3.7-flash: calling"),
            mock.call("Model gemini-3.7-flash: failed (empty response)"),
        ])

    @mock.patch("gemini_web2api.server.log")
    @mock.patch("gemini_web2api.server.generate")
    def test_combo_returns_first_successful_response(self, generate, log):
        generate.side_effect = [RuntimeError("denied"), "fallback ok"]
        _, attempts, _ = resolve_model_chain("gemini-combo")
        self.assertEqual(_generate_attempts("hello", attempts, None, True), "fallback ok")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(log.call_args_list[-2:], [
            mock.call("Model gemini-3.7-flash: calling"),
            mock.call("Model gemini-3.7-flash: success"),
        ])

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_combo_stream_does_not_switch_after_output(self, generate_stream):
        def partial_failure():
            yield "partial"
            raise RuntimeError("stream broke")

        generate_stream.side_effect = [partial_failure(), iter(["must not run"])]
        _, attempts, _ = resolve_model_chain("gemini-combo")
        stream = _generate_stream_attempts("hello", attempts, None, True)
        self.assertEqual(next(stream), "partial")
        with self.assertRaisesRegex(RuntimeError, "stream broke"):
            next(stream)
        self.assertEqual(generate_stream.call_count, 1)


class ConversationInputTests(unittest.TestCase):
    def tearDown(self):
        import gemini_web2api.server as server_module
        server_module._chat_history = []
        reset_conversation()

    @mock.patch("gemini_web2api.server.reset_conversation")
    def test_incremental_messages_send_only_new_user_turn(self, reset):
        first = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "first"},
        ]
        second = first + [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        ]

        self.assertEqual(_incremental_chat_messages(first), first)
        _commit_chat_messages(first)
        self.assertEqual(
            _incremental_chat_messages(second),
            [{"role": "user", "content": "follow up"}],
        )
        reset.assert_not_called()

    @mock.patch("gemini_web2api.server.reset_conversation")
    def test_shorter_history_resets_upstream_conversation(self, reset):
        old = [{"role": "user", "content": "old"}]
        _incremental_chat_messages(old)
        _commit_chat_messages(old)

        fresh = [{"role": "user", "content": "new"}]
        self.assertEqual(_incremental_chat_messages(fresh), fresh)

        reset.assert_called_once()

    @mock.patch("gemini_web2api.server.reset_conversation")
    def test_explicit_session_rebuilt_history_sends_only_latest_turn(self, reset):
        old = [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        rebuilt = [
            {"role": "system", "content": "new model system"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ]
        _commit_chat_messages(old, "selected-chat")

        self.assertEqual(
            _incremental_chat_messages(rebuilt, "selected-chat"),
            [{"role": "user", "content": "new question"}],
        )
        reset.assert_not_called()


class StreamingEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.original_config = dict(CONFIG)
        self.conversation_dir = tempfile.TemporaryDirectory()
        CONFIG["api_keys"] = []
        CONFIG["log_requests"] = False
        CONFIG["conversation_store_file"] = os.path.join(
            self.conversation_dir.name, "conversations.json"
        )
        import gemini_web2api.server as server_module
        server_module._chat_history = []
        server_module._chat_histories.clear()
        reset_conversation()
        self.refresh_auth_patcher = mock.patch(
            "gemini_web2api.server.refresh_auth",
            return_value={"cookie": "fake authenticated cookie"},
        )
        self.refresh_auth = self.refresh_auth_patcher.start()

    def tearDown(self):
        self.refresh_auth_patcher.stop()
        import gemini_web2api.server as server_module
        server_module._chat_history = []
        server_module._chat_histories.clear()
        reset_conversation()
        CONFIG.clear()
        CONFIG.update(self.original_config)
        self.conversation_dir.cleanup()

    def post_json(self, path, payload, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers=request_headers,
        )
        response = connection.getresponse()
        headers = dict(response.getheaders())
        chunks = []
        try:
            while chunk := response.read1(65536):
                chunks.append(chunk)
        except ConnectionResetError:
            if headers.get("Content-Type") != "text/event-stream":
                raise
        connection.close()
        return response.status, headers, b"".join(chunks).decode()

    def post_chunked_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    def get_json(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    def test_browser_conversation_manager_supports_empty_api_keys(self):
        status, headers, body = self.get_json("/conversations")

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn("Gemini Conversations", body)
        self.assertIn("leave empty when api_keys is []", body)

        status, _, body = self.post_json(
            "/v1/conversations", {"id": "browser-created"}
        )
        self.assertEqual(status, 201)
        self.assertTrue(json.loads(body)["active"])

    def test_chat_requires_model(self):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["message"], "model is required")

    def test_responses_requires_model(self):
        status, _, body = self.post_json("/v1/responses", {"input": "hello"})

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["message"], "model is required")

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_chat_stream_starts_with_assistant_role(self, generate_stream):
        generate_stream.return_value = iter(["hel", "lo"])

        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hel"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {"content": "lo"})
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate", return_value="chunked ok")
    def test_chat_accepts_chunked_body(self, _generate):
        status, _, body = self.post_chunked_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "chunked ok")

    @mock.patch("gemini_web2api.server.reset_conversation")
    @mock.patch("gemini_web2api.server.generate", return_value="continued")
    def test_explicit_conversation_survives_model_and_history_change(self, generate, reset):
        headers = {"X-Conversation-ID": "overlay-chat-1"}
        first = {
            "model": "gemini-3.6-flash",
            "messages": [{"role": "user", "content": "first"}],
        }
        changed = {
            "model": "gemini-auto",
            "messages": [{"role": "user", "content": "model changed"}],
        }

        first_status, first_headers, _ = self.post_json(
            "/v1/chat/completions", first, headers
        )
        second_status, second_headers, body = self.post_json(
            "/v1/chat/completions", changed, headers
        )

        self.assertEqual((first_status, second_status), (200, 200))
        self.assertEqual(first_headers["X-Conversation-ID"], "overlay-chat-1")
        self.assertEqual(second_headers["X-Conversation-ID"], "overlay-chat-1")
        self.assertEqual(json.loads(body)["conversation_id"], "overlay-chat-1")
        self.assertEqual([call.args[5] for call in generate.call_args_list], [
            "overlay-chat-1", "overlay-chat-1",
        ])
        reset.assert_not_called()

    @mock.patch("gemini_web2api.server.generate", return_value="active session")
    def test_selected_conversation_applies_without_custom_client_headers(self, generate):
        status, _, body = self.post_json(
            "/v1/conversations", {"id": "selected-chat"}
        )
        self.assertEqual(status, 201)
        self.assertTrue(json.loads(body)["active"])

        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "ordinary overlay request"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Conversation-ID"], "selected-chat")
        self.assertEqual(json.loads(body)["conversation_id"], "selected-chat")
        self.assertEqual(generate.call_args.args[5], "selected-chat")

    def test_conversation_attach_inspect_list_and_reset(self):
        status, headers, body = self.post_json(
            "/v1/conversations/site-chat/attach",
            {"cid": "gemini-cid", "rid": "gemini-rid", "rcid": "gemini-rcid"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Conversation-ID"], "site-chat")
        self.assertEqual(json.loads(body)["cid"], "gemini-cid")

        status, _, body = self.get_json("/v1/conversations/site-chat")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["metadata"][:3], [
            "gemini-cid", "gemini-rid", "gemini-rcid",
        ])
        status, _, body = self.get_json("/v1/conversations")
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in json.loads(body)["data"]], ["site-chat"])

        status, _, body = self.post_json("/v1/conversations/site-chat/reset", {})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["reset"])
        status, _, _ = self.get_json("/v1/conversations/site-chat")
        self.assertEqual(status, 404)

    def test_existing_conversation_can_be_selected(self):
        self.post_json("/v1/conversations", {"id": "first"})
        self.post_json("/v1/conversations", {"id": "second"})

        status, _, body = self.post_json("/v1/conversations/first/select", {})

        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["active"])
        self.assertEqual(gemini_module.active_conversation_id(), "first")

    def test_conversation_metadata_persists_after_memory_reload(self):
        auth = gemini_module.refresh_auth()
        create_conversation("persisted", ["cid", "rid", "rcid"], auth)
        self.assertTrue(os.path.isfile(CONFIG["conversation_store_file"]))

        gemini_module._conversation_states.clear()
        gemini_module._conversation_store_loaded = False

        self.assertEqual(
            conversation_info("persisted", auth)["metadata"][:3],
            ["cid", "rid", "rcid"],
        )

    def test_conversation_ids_isolate_payload_metadata(self):
        auth = gemini_module.refresh_auth()
        create_conversation("one", ["cid-1", "rid-1", "rcid-1"], auth)
        create_conversation("two", ["cid-2", "rid-2", "rcid-2"], auth)

        one = _decode_payload(_build_payload(
            "hello", 1, 4, auth=auth, conversation_id="one"
        ))
        two = _decode_payload(_build_payload(
            "hello", 1, 4, auth=auth, conversation_id="two"
        ))

        self.assertEqual(one[2][:3], ["cid-1", "rid-1", "rcid-1"])
        self.assertEqual(two[2][:3], ["cid-2", "rid-2", "rcid-2"])

    def test_attach_rejects_partial_gemini_metadata(self):
        status, _, body = self.post_json(
            "/v1/conversations/broken/attach", {"cid": "only-cid"}
        )

        self.assertEqual(status, 400)
        self.assertIn("cid, rid, and rcid", json.loads(body)["error"]["message"])

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="looks good")
    def test_chat_accepts_openai_image_url_data_url(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], [("/uploaded/image-ref", "image.png")])
        self.assertIn("[Image attached]", generate.call_args.args[0])
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "looks good")

    @mock.patch("gemini_web2api.server.upload_image")
    @mock.patch("gemini_web2api.server.generate")
    def test_chat_rejects_anonymous_image_before_upload(self, generate, upload_image):
        self.refresh_auth.return_value = {"cookie": ""}
        image_data = base64.b64encode(b"same image").decode()

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe it"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    ],
                }],
            },
        )

        self.assertEqual(status, 400)
        self.assertIn("requires authenticated Gemini cookies", json.loads(body)["error"]["message"])
        upload_image.assert_not_called()
        generate.assert_not_called()

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/unique-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="deduplicated")
    def test_chat_uploads_only_latest_history_image(self, generate, upload_image):
        old_image_data = base64.b64encode(b"old image").decode()
        new_image_data = base64.b64encode(b"new image").decode()
        old_image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{old_image_data}"},
        }
        new_image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{new_image_data}"},
        }

        status, _, _ = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "First"}, old_image_part]},
                    {"role": "assistant", "content": "Earlier answer"},
                    {"role": "user", "content": [{"type": "text", "text": "Again"}, new_image_part]},
                ],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"new image", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], [("/uploaded/unique-ref", "image.png")])

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", side_effect=GeminiUpstreamError(1100))
    def test_chat_reports_expired_image_session_as_auth_error(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe it"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    ],
                }],
            },
        )

        self.assertEqual(status, 401)
        message = json.loads(body)["error"]["message"]
        self.assertIn("expired or unauthenticated", message)
        self.assertIn("export a new gemini-auth.json", message)
        upload_image.assert_called_once()
        generate.assert_called_once()

    @mock.patch("gemini_web2api.server.fetch_image_bytes", return_value=b"\xff\xd8\xffremote jpeg")
    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/remote-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="remote ok")
    def test_responses_accepts_input_image_url(self, generate, upload_image, fetch_image_bytes):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is shown?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        fetch_image_bytes.assert_called_once_with("https://example.com/image.jpg")
        upload_image.assert_called_once_with(b"\xff\xd8\xffremote jpeg", "image.png", "image/jpeg")
        self.assertEqual(generate.call_args.args[3], [("/uploaded/remote-ref", "image.png")])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="top-level image ok")
    def test_responses_accepts_top_level_input_image(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [
                    {"type": "input_text", "text": "What is shown?"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    },
                ],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], [("/uploaded/image-ref", "image.png")])
        self.assertIn("What is shown?", generate.call_args.args[0])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", side_effect=RuntimeError("upload denied"))
    def test_google_image_upload_failure_returns_502(self, _upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:generateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_data,
                        },
                    }],
                }],
            },
        )

        self.assertEqual(status, 502)
        self.assertIn("image upload failed: upload denied", json.loads(body)["error"]["message"])

    @mock.patch("gemini_web2api.server.generate_stream", return_value=iter(["streamed"]))
    def test_google_stream_generate_content_uses_sse(self, _generate_stream):
        status, headers, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:streamGenerateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": "Stream this"}],
                }],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn('"text": "streamed"', body)

    @mock.patch("gemini_web2api.server.generate", return_value="hello")
    def test_responses_text_stream_has_complete_event_sequence(self, _generate):
        status, headers, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "hello",
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[4][1]["delta"], "hello")
        self.assertEqual(events[-1][1]["response"]["status"], "completed")
        self.assertEqual(events[-1][1]["response"]["output"][0]["content"][0]["text"], "hello")

    @mock.patch("gemini_web2api.server.parse_tool_calls")
    @mock.patch("gemini_web2api.server.generate", return_value="tool output")
    def test_responses_function_call_stream_has_complete_event_sequence(
        self, _generate, parse_tool_calls
    ):
        parse_tool_calls.return_value = (
            "",
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Shanghai"}'},
                }
            ],
        )

        status, _, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "weather",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[2][1]["output_index"], 0)
        self.assertEqual(events[3][1]["delta"], '{"city":"Shanghai"}')
        self.assertEqual(events[4][1]["arguments"], '{"city":"Shanghai"}')
        self.assertEqual(events[-1][1]["response"]["output"][0]["name"], "get_weather")


if __name__ == "__main__":
    unittest.main()
