import unittest
from unittest import mock

import server


class DeploymentRoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_health_is_ready_even_without_a_loaded_corpus(self):
        with mock.patch.object(server.engine, 'ready', False):
            response = self.client.get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'status': 'ok'})

    def test_frontend_assets_and_legacy_front_prefix_still_work(self):
        for path in ('/', '/static.css', '/app/frontend/app.js',
                     '/app/frontend/styles.css', '/app/static/logo2.png',
                     '/app/static/logo3.png', '/front/static.css'):
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)

    def test_server_configuration_and_cache_are_not_public_files(self):
        for path in ('/server.py', '/requirements.txt', '/gunicorn.conf.py',
                     '/.git/config', '/.cache/dense/meta.json',
                     '/front/../server.py', '/app/frontend/../../../server.py'):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == '__main__':
    unittest.main()
