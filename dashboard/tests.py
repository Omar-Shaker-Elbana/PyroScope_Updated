import json
import os
from unittest import mock

from django.test import TestCase

from core.src.config import resolve_firms_key, set_user_firms_key

KEY = "0123456789abcdef0123456789abcdef"
URL = "/api/dashboard/firms-key/"


class FirmsKeyApiTests(TestCase):
    def setUp(self):
        set_user_firms_key(None)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(set_user_firms_key, None)
        os.environ.pop("FIRMS_MAP_KEY", None)

    def post(self, key):
        return self.client.post(URL, data=json.dumps({"key": key}), content_type="application/json")

    def test_user_key_is_used_when_env_is_missing(self):
        r = self.post(KEY)
        self.assertEqual(r.json()["active"], "user")
        self.assertEqual(resolve_firms_key(), KEY)
        self.assertNotIn(KEY, r.content.decode())            # never echoed back

    def test_env_key_wins_and_user_key_is_not_kept(self):
        os.environ["FIRMS_MAP_KEY"] = "ENVKEY" + KEY
        r = self.post(KEY)
        self.assertEqual(r.json()["active"], "env")
        self.assertFalse(r.json()["user_key"])
        self.assertEqual(resolve_firms_key(), "ENVKEY" + KEY)

    def test_malformed_key_is_rejected(self):
        r = self.post("abc/../../x?y=1")
        self.assertEqual(r.status_code, 400)
        self.assertIsNone(resolve_firms_key())

    def test_delete_forgets_the_key(self):
        self.post(KEY)
        r = self.client.delete(URL)
        self.assertIsNone(r.json()["active"])
        self.assertIsNone(resolve_firms_key())
