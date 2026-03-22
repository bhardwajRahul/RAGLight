import subprocess
import tempfile
from pathlib import Path
from typing import List
import logging
from ..models.data_source_model import GitHubSource


class GithubScrapper:
    """
    A utility class for cloning and managing GitHub repositories.

    Attributes:
        repositories (List[GitHubSource]): A list of GitHub repository sources to manage.
    """

    def __init__(self) -> None:
        self.repositories: List[GitHubSource] = []

    @staticmethod
    def clone_github_repo(repo_url: str, clone_path: str, branch: str = "main") -> None:
        """
        Clones a GitHub repository to a specified local directory.

        Args:
            repo_url (str): The URL of the GitHub repository to clone.
            clone_path (str): The local directory where the repository will be cloned.
            branch (str, optional): The branch of the repository to clone. Defaults to "main".

        Raises:
            subprocess.CalledProcessError: If the git command fails.
        """
        command = ["git", "clone", "--branch", branch, repo_url, clone_path]
        logging.info(f"Executing: {' '.join(command)}")
        subprocess.run(command, check=True)
        logging.info("✅ Clone successful!")

    def clone_all(self) -> str:
        """
        Clones all repositories in the `repo_urls` list to a temporary directory.

        Args:
            branch (str, optional): The branch of each repository to clone. Defaults to "main".

        Returns:
            str: The path to the temporary directory containing the cloned repositories.
        """
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        logging.info(f"Cloning repositories into folder: {temp_path}")

        for repo in self.repositories:
            url = repo.url
            branch = repo.branch if hasattr(repo, "branch") else "main"
            repo_name = url.split("/")[-1].replace(".git", "")
            clone_path = temp_path / repo_name
            try:
                self.clone_github_repo(url, str(clone_path), branch)
            except Exception as e:
                logging.error(f"Failed to clone {url}: {e}")

        return str(temp_path)

    def get_urls(self) -> List[str]:
        """
        Retrieves the list of repository URLs managed by the GithubScrapper.

        Returns:
            List[str]: The list of repository URLs.
        """
        return [self.repositories[i].url for i in range(len(self.repositories))]

    def set_repositories(self, repositories: List[GitHubSource]) -> None:
        """
        Sets the list of GitHub repositories to manage.

        Args:
            repositories (List[GitHubSource]): A list of GitHub repository sources to manage.
        """
        self.repositories = repositories
