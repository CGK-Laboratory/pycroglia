# Project

## Issue association

All tasks, features, bug fixes, and TODOs must have an associated Issue in the repository. Every code change, pull request, or TODO comment should reference its corresponding Issue by number (e\.g\., `#42`\). This ensures traceability, clear context, and proper project tracking.

- Do not merge code that is not linked to an Issue.
- Use the Issue number in branch names, commit messages, PRs, and TODO comments.
- Example TODO comment: `# TODO(#42): Refactor authentication logic for clarity.`

## Creating Issues

Follow these best practices when creating a new Issue:

- Use clear, descriptive titles summarizing the problem or proposal.
- Provide detailed context, steps to reproduce (for bugs), or acceptance criteria (for features).
- Use GitHub Issue templates if available.
- Assign appropriate labels (e\.g\., `bug`, `enhancement`, `documentation`\).
- Link related Issues or PRs when relevant.
- Assign the Issue to yourself or the responsible contributor if known.
- Keep the Issue updated with progress, discussions, and resolutions.

Refer to the [GitHub Issues documentation](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) for more details.

## Branch naming conventions

Branch naming conventions: When naming branches, use the following prefixes based on the type of work being done, as specified by the labels in your issue tracker:

- `bug/issueID-short-description`: For branches fixing a bug.

- `feat/issueID-short-description`: For branches introducing new features or enhancements.

- `test/issueID-short-description`: For branches related to testing tasks and development of new tests.

- `change/issueID-short-description`: For branches dedicated to applying a change to current code that do not fix bugs or add 
improvements, such as rollbacks.

Make sure to replace issueID with the corresponding issue identifier and provide a short description that summarizes the 
purpose of the branch. Follow standard conventions for branch names, such as using lowercase letters, numbers, hyphens, 
and underscores, but avoiding spaces or special characters.

## Commit Message Guidelines

- Use the imperative mood in the subject line (e\.g\., `Add new feature` instead of `Added` or `Adds`\.)
- Limit the subject line to 50 characters or less\.
- Capitalize the first letter of the subject line\.
- Do not end the subject line with a period\.
- Separate the subject from the body with a blank line\.
- Use the body to explain _what_ and _why_ vs\. _how_ \(wrap at 72 characters\)\.
- Reference issues and pull requests when relevant \(e\.g\., `Closes #123`\)\.
- Use concise and descriptive language\.
- Group related changes in a single commit; avoid unrelated changes\.

### Examples

- `Fix typo in README`
- `Add user authentication`
- `Update dependencies to latest versions`
- `Refactor data processing module`
- `Remove deprecated API endpoints`
- `Improve error handling in login flow`
- `Document configuration options`
- `Test edge cases for payment processing`

## Pull request guidelines

- Create a PR for each Issue. One issue must be associated to only 1 PR.
- Open the PR as a **Draft**. Once all changes are complete and ready for review, mark the PR as ready for review.
- Reference related issue in your PR description (`Closes #12`).
- Ensure all new code is covered by tests.
- Run all checks before pushing. If no CI/CD is available when you create the PR, 
the results of the checks must be attached to the PR:

```bash
  uv run ruff check
  uv run ruff format
  uv run pytest
```

# Python
## Docstrings

This project uses the Google format described [here](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings) 
for the docstring formats. Here is an example of the expected docstring format:

```python
def calculate_sum(a, b):
    """Calculates the sum of two numbers.

    This function takes two numerical inputs and returns their sum.

    Args:
        a (int or float): The first number.
        b (int or float): The second number.

    Returns:
        int or float: The sum of a and b.

    Raises:
        TypeError: If a or b are not numerical types.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numerical.")
    return a + b
```

## Linter and formatter

This project uses Ruff as the default formatter and linter. The rules used can be found in the `pyproject.toml` file.

Linter command:
```shell
uv run ruff check
```

Formatter command:
```shell
uv run ruff format
```
