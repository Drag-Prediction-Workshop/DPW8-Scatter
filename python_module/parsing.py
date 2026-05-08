import re
from pathlib import Path

from load_data import REPO_ROOT, find_datafiles, parse_datafile_metadata


VARIABLES_RE = re.compile(r"^VARIABLES\s*=\s*(?P<rest>.*)$", re.IGNORECASE)
TITLE_RE = re.compile(r'^TITLE\s*=\s*"(?P<title>[^"]*)"', re.IGNORECASE)
ZONE_RE = re.compile(r'^ZONE\b.*\bT\s*=\s*"(?P<title>[^"]*)"', re.IGNORECASE)
DATASETAUX_RE = re.compile(r"^DATASETAUXDATA\b", re.IGNORECASE)
ZONEAUX_RE = re.compile(r"^AUXDATA\b", re.IGNORECASE)
NUMERIC_ROW_RE = re.compile(r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?(?:\s|$)")


def _extract_ids_from_path(path):
    path = Path(path)
    parts = path.parts
    if len(parts) < 3:
        return None

    try:
        participant = parts[-3].split("_")[0]
        submission = parts[-2].split("_")[0]
    except Exception:
        return None

    if participant.isnumeric() and submission.isnumeric():
        return f"{participant}.{submission}"
    return None


def _parse_variables(lines, start_index):
    match = VARIABLES_RE.match(lines[start_index].strip())
    if match is None:
        return [], start_index

    payload = match.group("rest").strip()
    index = start_index
    if not payload and start_index + 1 < len(lines):
        index = start_index + 1
        payload = lines[index].strip()

    return re.findall(r'"([^"]+)"', payload), index


def _line_is_zone_info(line):
    if "=" not in line:
        return False
    for item in line.split(","):
        if "=" not in item:
            return False
    return True


def _analyze_ascii_file(path):
    path = Path(path).expanduser().resolve()
    lines = path.read_text(errors="replace").splitlines()
    variables = []
    zones = []
    zone_row_counts = []
    current_zone_index = -1
    unknown_lines = []
    row_count_mismatches = []
    data_before_zone_count = 0
    title = ""

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        title_match = TITLE_RE.match(line)
        if title_match:
            title = title_match.group("title")
            continue

        if VARIABLES_RE.match(line):
            variables, i2 = _parse_variables(lines, i - 1)
            i = i2 + 1
            continue

        zone_match = ZONE_RE.match(line)
        if zone_match:
            zones.append(zone_match.group("title"))
            zone_row_counts.append(0)
            current_zone_index += 1
            continue

        if DATASETAUX_RE.match(line):
            continue

        if ZONEAUX_RE.match(line):
            continue

        if NUMERIC_ROW_RE.match(line):
            if current_zone_index < 0:
                data_before_zone_count += 1
                continue
            row = line.split()
            zone_row_counts[current_zone_index] += 1
            if variables and len(row) != len(variables):
                row_count_mismatches.append(
                    {
                        "zone": zones[current_zone_index],
                        "expected": len(variables),
                        "found": len(row),
                        "line": i,
                    }
                )
            continue

        if _line_is_zone_info(line):
            continue

        unknown_lines.append({"line": i, "text": raw_line})

    return {
        "path": path,
        "title": title,
        "variables": variables,
        "zones": zones,
        "zone_row_counts": zone_row_counts,
        "unknown_lines": unknown_lines,
        "row_count_mismatches": row_count_mismatches,
        "data_before_zone_count": data_before_zone_count,
    }


def _validate_analysis(analysis):
    errors = []
    warnings = []

    if not analysis["title"]:
        errors.append("missing TITLE")
    if not analysis["variables"]:
        errors.append("missing VARIABLES list")
    if not analysis["zones"]:
        errors.append("missing ZONE entries")
    if analysis["data_before_zone_count"] > 0:
        errors.append(
            f"found {analysis['data_before_zone_count']} numeric row(s) before first ZONE"
        )
    if analysis["row_count_mismatches"]:
        errors.append(
            f"found {len(analysis['row_count_mismatches'])} data row(s) with variable-count mismatch"
        )
    if analysis["unknown_lines"]:
        warnings.append(
            f"found {len(analysis['unknown_lines'])} line(s) that do not match known Tecplot ASCII patterns"
        )

    expected_title = _extract_ids_from_path(analysis["path"])
    if expected_title and analysis["title"] and analysis["title"] != expected_title:
        errors.append(
            f"title '{analysis['title']}' does not match path-derived id '{expected_title}'"
        )

    return errors, warnings


def _summarize_result(path, metadata, analysis, errors, warnings):
    return {
        "path": str(path),
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metadata_numeric_rows": metadata["numeric_rows"],
        "parsed_zone_count": len(analysis["zones"]),
        "parsed_variable_count": len(analysis["variables"]),
        "zone_row_counts": analysis["zone_row_counts"],
        "row_count_mismatches": analysis["row_count_mismatches"],
        "unknown_lines": analysis["unknown_lines"],
    }


def _print_result(result):
    status = "OK" if result["ok"] else "ERROR"
    print(f"[{status}] {result['path']}")
    if result["errors"]:
        for item in result["errors"]:
            print(f"  - error: {item}")
    if result["warnings"]:
        for item in result["warnings"]:
            print(f"  - warning: {item}")


def universal_parser(
    paths=None,
    *,
    test_cases=None,
    file_pattern="*.dat",
    root=REPO_ROOT,
    verbose=False,
    strict=False,
    max_files=None,
):
    if paths is None:
        candidates = find_datafiles(
            test_cases=test_cases,
            file_pattern=file_pattern,
            root=root,
        )
    else:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        candidates = [Path(path).expanduser().resolve() for path in paths]

    if max_files is not None:
        candidates = candidates[: int(max_files)]

    results = []
    for path in candidates:
        metadata = parse_datafile_metadata(path)
        analysis = _analyze_ascii_file(path)
        errors, warnings = _validate_analysis(analysis)

        if metadata["numeric_rows"] == 0:
            errors.append("no numeric data rows found")

        result = _summarize_result(path, metadata, analysis, errors, warnings)
        results.append(result)
        if verbose:
            _print_result(result)

    failures = [result for result in results if not result["ok"]]
    summary = {
        "total_files": len(results),
        "ok_files": len(results) - len(failures),
        "failed_files": len(failures),
        "results": results,
    }

    if strict and failures:
        first = failures[0]
        raise RuntimeError(
            f"universal_parser found {len(failures)} invalid file(s). "
            f"First failure: {first['path']} :: {first['errors'][0]}"
        )

    return summary
