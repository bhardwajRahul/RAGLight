import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from raglight.scrapper.github_scrapper import GithubScrapper
from raglight.models.data_source_model import GitHubSource


class TestGithubScrapper(unittest.TestCase):
    def test_init_repositories_is_empty_list(self):
        scrapper = GithubScrapper()
        self.assertEqual(scrapper.repositories, [])

    def test_clone_all_without_set_repositories_does_not_raise(self):
        scrapper = GithubScrapper()
        path = scrapper.clone_all()
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.isdir(path))
        shutil.rmtree(path, ignore_errors=True)

    def test_set_repositories_then_get_urls(self):
        scrapper = GithubScrapper()
        repos = [
            GitHubSource(url="https://github.com/user/repo1"),
            GitHubSource(url="https://github.com/user/repo2"),
        ]
        scrapper.set_repositories(repos)
        self.assertEqual(scrapper.get_urls(), [
            "https://github.com/user/repo1",
            "https://github.com/user/repo2",
        ])

    @patch("raglight.scrapper.github_scrapper.subprocess.run")
    def test_clone_all_calls_git_for_each_repo(self, mock_run):
        scrapper = GithubScrapper()
        scrapper.set_repositories([
            GitHubSource(url="https://github.com/user/repo1"),
            GitHubSource(url="https://github.com/user/repo2"),
        ])
        path = scrapper.clone_all()
        self.assertEqual(mock_run.call_count, 2)
        shutil.rmtree(path, ignore_errors=True)

    @patch("raglight.scrapper.github_scrapper.subprocess.run", side_effect=Exception("network error"))
    def test_clone_all_continues_on_failure(self, mock_run):
        scrapper = GithubScrapper()
        scrapper.set_repositories([
            GitHubSource(url="https://github.com/user/repo1"),
            GitHubSource(url="https://github.com/user/repo2"),
        ])
        # should not raise, returns path even if all clones fail
        path = scrapper.clone_all()
        self.assertIsInstance(path, str)
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
