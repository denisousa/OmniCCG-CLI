from urllib.parse import urlparse
from pathlib import Path
import click

def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def write_xml_result(lineages_xml, metrics_xml, result_path: str | None = None):
    """
    Save XML contents to files:
      - ccg_data.xml
      - ccg_metrics.xml
    If result_path is provided, files are written there; otherwise to CWD.
    """
    out_dir = Path(result_path) if result_path else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_lineages = out_dir / "ccg_data.xml"
    dst_metrics = out_dir / "ccg_metrics.xml"

    dst_lineages.write_text(lineages_xml, encoding="utf-8")
    dst_metrics.write_text(metrics_xml, encoding="utf-8")

    click.echo(f"Wrote: {dst_lineages}")
    click.echo(f"Wrote: {dst_metrics}")


def enforce_single_selector(user_settings: dict) -> None:
    """
    Enforce that exactly one among:
      - from_first_commit (bool)
      - from_a_specific_commit (non-empty str)
      - days_prior (int > 0)
    is selected. If none provided, default to from_first_commit=True.
    """
    ffc = bool(user_settings.get("from_first_commit"))
    fac = user_settings.get("from_a_specific_commit")
    dp = user_settings.get("days_prior")

    has_fac = isinstance(fac, str) and fac.strip() != ""
    has_dp = isinstance(dp, int) and dp > 0

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
