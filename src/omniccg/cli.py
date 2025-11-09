import click
import json
from .core import execute_omniccg

@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to config JSON file'
)
@click.option('--git-repo', '-g', help='Git repository URL')
@click.option('--from-first-commit', is_flag=True,
              help='Start extraction from the first commit')
@click.option('--from-commit', help='Start extraction from a specific commit (hash)')
@click.option('--days-prior', type=int, help='Analyze commits from X days prior')
@click.option('--merge-commit', help='Analyze a specific merge commit')
@click.option('--fixed-leaps', type=int, help='Fixed number of commits to leap')
@click.option('--clone-detector', default='nicad',
              help='Clone detector to use (default: nicad)')
@click.option('--language', default='java',
              help='Programming language (default: java)')
def main(config, git_repo, from_first_commit, from_commit, days_prior,
         merge_commit, fixed_leaps, clone_detector, language):
    """OmniCCG CLI — Code Clone Genealogy Tool"""

    # 1) If a config file is provided, use it and ignore other flags
    if config:
        with open(config, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # Minimal validation
        if not isinstance(settings, dict) or 'git_repository' not in settings:
            raise click.UsageError(
                "The configuration JSON must contain the 'git_repository' key."
            )

        # Pass the file dictionary as-is
        try:
            return execute_omniccg(settings)
        except ValueError as e:
            raise click.UsageError(str(e))

    # 2) Without a file: build a dictionary equivalent to the expected JSON
    if not git_repo:
        raise click.UsageError('Git repository URL is required (use --git-repo or --config).')

    settings = {
        "git_repository": git_repo,
        "user_settings": {
            "from_first_commit": bool(from_first_commit),
            "from_a_specific_commit": from_commit,   # may be None
            "days_prior": days_prior,                # may be None
            "merge_commit": merge_commit,            # may be None
            "fixed_leaps": fixed_leaps,              # may be None
            "clone_detector": clone_detector,
            "language": language
        }
    }

    try:
        return execute_omniccg(settings)
    except ValueError as e:
        raise click.UsageError(str(e))


if __name__ == '__main__':
    main()
