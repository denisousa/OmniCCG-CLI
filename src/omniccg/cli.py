import json
from urllib.parse import urlparse
import click
from .core import execute_omniccg


def _is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _enforce_single_selector(user_settings: dict) -> None:
    """
    Enforce that exactly one among:
      - from_first_commit (bool)
      - from_a_specific_commit (non-empty str)
      - days_prior (int > 0)
    is selected. If none provided, default to from_first_commit=True.
    """
    ffc = bool(user_settings.get("from_first_commit"))
    fac = user_settings.get("from_a_specific_commit")
    dp  = user_settings.get("days_prior")

    has_fac = isinstance(fac, str) and fac.strip() != ""
    has_dp  = isinstance(dp, int) and dp > 0

    count = int(ffc) + int(has_fac) + int(has_dp)

    if count == 0:
        # default: from_first_commit = True
        user_settings["from_first_commit"] = True
        user_settings["from_a_specific_commit"] = None
        user_settings["days_prior"] = None
        return

    if count > 1:
        raise click.UsageError(
            "Select only ONE of: from_first_commit OR from_a_specific_commit OR days_prior."
        )

    # Normalize the non-selected ones to None/False for clarity
    if ffc:
        user_settings["from_a_specific_commit"] = None
        user_settings["days_prior"] = None
        user_settings["from_first_commit"] = True
    elif has_fac:
        user_settings["from_first_commit"] = False
        user_settings["days_prior"] = None
    elif has_dp:
        user_settings["from_first_commit"] = False
        user_settings["from_a_specific_commit"] = None


@click.command()
@click.option("--config", "-c",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to config JSON file")
@click.option("--git-repo", "-g", help="Git repository URL")
@click.option("--from-first-commit", is_flag=True, help="Start from the first commit (default)")
@click.option("--from-commit", help="Start from a specific commit (hash)")
@click.option("--days-prior", type=int, help="Analyze commits from N days prior")
@click.option("--merge-commit", help="Analyze a specific merge commit")  # default: not used
@click.option("--fixed-leaps", type=int, help="Fixed number of commits to leap")  # default: not used
@click.option("--clone-detector", default="nicad",
              help="Built-in clone detector to use when 'detection-api' is absent (default: nicad)")
@click.option("--detection-api",
              help="HTTP endpoint of the external detection API; if set, 'clone_detector' is ignored")
def main(config, git_repo, from_first_commit, from_commit, days_prior,
         merge_commit, fixed_leaps, clone_detector, detection_api):
    """OmniCCG CLI — enforce single selection; default from_first_commit=True; optional detection-api."""

    # --- 1) Config file path provided ---
    if config:
        with open(config, "r", encoding="utf-8") as f:
            settings = json.load(f)

        if not isinstance(settings, dict) or "git_repository" not in settings:
            raise click.UsageError("Config JSON must contain 'git_repository'.")

        us = settings.setdefault("user_settings", {})

        # Enforce single selector (default to from_first_commit=True)
        _enforce_single_selector(us)

        # detection-api logic
        det_api = settings.get("detection-api")
        if det_api is not None:
            if not isinstance(det_api, str) or not _is_valid_url(det_api):
                raise click.UsageError("When present, 'detection-api' must be a valid http(s) URL string.")
            # ignore clone_detector when detection-api present
            if "clone_detector" in us:
                us.pop("clone_detector", None)
                click.echo("Notice: 'detection-api' provided — 'clone_detector' will be ignored.", err=True)
        else:
            # no detection-api: ensure clone_detector present (default nicad)
            us.setdefault("clone_detector", "nicad")

        # By default, do not use leaps or merge unless provided in config
        us.setdefault("merge_commit", None)
        us.setdefault("fixed_leaps", None)

        try:
            return execute_omniccg(settings)
        except ValueError as e:
            raise click.UsageError(str(e))

    # --- 2) No config file: build from CLI flags ---
    if not git_repo:
        raise click.UsageError("Git repository URL is required (use --git-repo or --config).")

    # Build user_settings from CLI
    us = {
        # selectors (we'll enforce mutual exclusivity below)
        "from_first_commit": bool(from_first_commit),
        "from_a_specific_commit": from_commit,
        "days_prior": days_prior,

        # defaults: not used unless explicitly passed
        "merge_commit": merge_commit,
        "fixed_leaps": fixed_leaps,
    }

    # Enforce single selector; default to from_first_commit=True if none given
    _enforce_single_selector(us)

    settings = {
        "git_repository": git_repo,
        "user_settings": us
    }

    # detection-api (CLI) takes precedence; do NOT set clone_detector if present
    if detection_api:
        if not _is_valid_url(detection_api):
            raise click.UsageError("Please provide a valid --detection-api (http/https).")
        settings["detection-api"] = detection_api
        click.echo("Notice: --detection-api provided — '--clone-detector' will be ignored.", err=True)
    else:
        settings["user_settings"]["clone_detector"] = clone_detector

    try:
        return execute_omniccg(settings)
    except ValueError as e:
        raise click.UsageError(str(e))


if __name__ == "__main__":
    main()
