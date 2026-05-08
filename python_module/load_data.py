import re
from pathlib import Path

import tecplot as tp
from tecplot.constant import (
    AnchorAlignment,
    Color,
    FillMode,
    GeomShape,
    LegendShow,
    PlotType,
    ReadDataOption,
    SymbolType,
    TextBox,
    Units,
)
from tecplot.exception import TecplotLogicError, TecplotSystemError


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_CASES = None
DEFAULT_DATAFILE_PATTERN = "*ForceMoment*.dat"
DEFAULT_ALPHA = 1.50

FORCE_MOMENT_VARIABLES = [
    "GRID_LEVEL",
    "GRID_SIZE",
    "GRID_FAC",
    "MACH",
    "REY",
    "ALPHA",
    "CL_TOT",
    "CD_TOT",
    "CM_TOT",
    "CL_WING",
    "CD_WING",
    "CM_WING",
    "CD_PR",
    "CD_SF",
    "CL_TAIL",
    "CD_TAIL",
    "CM_TAIL",
    "CL_FUS",
    "CD_FUS",
    "CM_FUS",
    "CL_NAC",
    "CD_NAC",
    "CM_NAC",
    "CL_PY",
    "CD_PY",
    "CM_PY",
    "CPU_Hours",
    "DELTAT",
    "CTUSTART",
    "CTUAVG",
    "Q/E",
]

VARIABLE_DISPLAY_NAMES = {
    "ALPHA": "AoA",
    "CL_TOT": "CL<sub>Total</sub>",
    "CD_TOT": "CD<sub>Total</sub>",
    "CM_TOT": "CMy<sub>Total</sub>",
    "CL_WING": "CL<sub>Wing</sub>",
    "CD_WING": "CD<sub>Wing</sub>",
    "CM_WING": "CMy<sub>Wing</sub>",
    "CD_PR": "CD<sub>Pressure</sub>",
    "CD_SF": "CD<sub>SkinFriction</sub>",
    "CL_TAIL": "CL<sub>Tail</sub>",
    "CD_TAIL": "CD<sub>Tail</sub>",
    "CM_TAIL": "CMy<sub>Tail</sub>",
    "CL_FUS": "CL<sub>Fuselage</sub>",
    "CD_FUS": "CD<sub>Fuselage</sub>",
    "CM_FUS": "CMy<sub>Fuselage</sub>",
    "CL_NAC": "CL<sub>Nacelle</sub>",
    "CD_NAC": "CD<sub>Nacelle</sub>",
    "CM_NAC": "CMy<sub>Nacelle</sub>",
    "CL_PY": "CL<sub>Pylon</sub>",
    "CD_PY": "CD<sub>Pylon</sub>",
    "CM_PY": "CMy<sub>Pylon</sub>",
    "CPU_Hours": "CPU<sub>Hours</sub>",
    "DELTAT": "<greek>D</greek>T",
    "CTUSTART": "CTU<sub>Start</sub>",
    "CTUAVG": "CTU<sub>Avg</sub>",
}

DERIVED_VARIABLE_EQUATIONS = [
    "{h = N<sup>-1</sup>} = 1/({GRID_SIZE}**(1/1))",
    "{h = N<sup>-1/2</sup>} = 1/({GRID_SIZE}**(1/2))",
    "{h = N<sup>-1/3</sup>} = 1/({GRID_SIZE}**(1/3))",
    "{h = N<sup>-2/3</sup>} = 1/({GRID_SIZE}**(2/3))",
]

LINE_COLORS = [
    Color.Red,
    Color.Green,
    Color.Blue,
    Color.Purple,
    Color.Orange,
    Color.Cinnamon,
    Color.SkyBlue,
    Color.Custom15,
    Color.Custom29,
    Color.Custom32,
    Color.DeepRed,
    Color.Forest,
    Color.RoyalBlue,
    Color.DeepViolet,
    Color.Turquoise,
    Color.Magenta,
]

SYMBOL_PRESETS = (
    GeomShape.Square,
    GeomShape.Del,
    GeomShape.Circle,
    GeomShape.Diamond,
)

TITLE_RE = re.compile(r'^TITLE\s*=\s*"(?P<title>[^"]*)"', re.IGNORECASE)
DATASET_AUX_RE = re.compile(
    r'^DATASETAUXDATA\s+(?P<key>[A-Za-z0-9_.]+)\s*=\s*(?P<value>.*)$',
    re.IGNORECASE,
)
ZONE_RE = re.compile(r'^ZONE\b.*\bT\s*=\s*"(?P<title>[^"]*)"', re.IGNORECASE)
ZONE_AUX_RE = re.compile(
    r'^AUXDATA\s+(?P<key>[A-Za-z0-9_.]+)\s*=\s*(?P<value>.*)$',
    re.IGNORECASE,
)
NUMERIC_ROW_RE = re.compile(
    r'^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?(?:\s|$)'
)
ALPHA_ZONE_RE = re.compile(r'\bALPHA\s+([+-]?\d+(?:\.\d+)?)\b', re.IGNORECASE)
GRID_ZONE_RE = re.compile(r'\bGRID\s+LEVEL\s+([+-]?\d+)\b', re.IGNORECASE)


def clear_existing_linemaps(frame):
    """Clear current linemaps."""
    plot = frame.plot(PlotType.XYLine)
    plot.activate()
    plot.delete_linemaps()


def recreate_linemaps(frame, x_var, y_var):
    dataset = frame.dataset
    plot = frame.plot(PlotType.XYLine)
    plot.activate()
    linemaps = []

    for zone_index in range(dataset.num_zones):
        zone = dataset.zone(zone_index)
        linemap = plot.add_linemap(zone.name[:6], zone=zone, x=x_var, y=y_var)
        linemap.show = True

    return linemaps


def clean_aux_value(value):
    return value.strip().strip('"')


def parse_datafile_metadata(path):
    path = Path(path).expanduser().resolve()
    metadata = {
        "path": path,
        "title": "",
        "aux": {},
        "zones": [],
        "numeric_rows": 0,
    }
    current_zone = None

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = TITLE_RE.match(line)
        if match:
            metadata["title"] = match.group("title")
            continue

        match = DATASET_AUX_RE.match(line)
        if match:
            metadata["aux"][match.group("key")] = clean_aux_value(match.group("value"))
            continue

        match = ZONE_RE.match(line)
        if match:
            current_zone = {
                "name": match.group("title"),
                "aux": {},
                "numeric_rows": 0,
            }
            metadata["zones"].append(current_zone)
            continue

        match = ZONE_AUX_RE.match(line)
        if match and current_zone is not None:
            current_zone["aux"][match.group("key")] = clean_aux_value(
                match.group("value")
            )
            continue

        if current_zone is not None and NUMERIC_ROW_RE.match(line):
            current_zone["numeric_rows"] += 1
            metadata["numeric_rows"] += 1

    return metadata


def find_datafiles(
    test_cases=DEFAULT_TEST_CASES,
    file_pattern=DEFAULT_DATAFILE_PATTERN,
    root=REPO_ROOT,
):
    """Find data files under TestCase directories."""
    root = Path(root).expanduser().resolve()

    if test_cases is None:
        test_case_paths = sorted(
            path for path in root.glob("TestCase*") if path.is_dir()
        )
    else:
        if isinstance(test_cases, (str, Path)):
            test_cases = [test_cases]

        test_case_paths = []
        for test_case in test_cases:
            path = Path(test_case).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()

            if not path.is_dir():
                matches = sorted(
                    child
                    for child in root.iterdir()
                    if (
                        child.is_dir()
                        and child.name.lower() == Path(test_case).name.lower()
                    )
                )
                if matches:
                    path = matches[0]

            if not path.is_dir():
                print(f"could not find test case directory at {path}")
                continue

            test_case_paths.append(path)

    datafiles = []
    for test_case_path in test_case_paths:
        datafiles.extend(
            path
            for path in test_case_path.rglob(file_pattern)
            if path.is_file()
        )

    return sorted(datafiles)


def load_single_datafile(path, replace_existing=False):
    """Load one Tecplot data file into the active frame."""
    frame = tp.active_frame()
    path = Path(path).expanduser().resolve()

    return tp.data.load_tecplot(
        str(path),
        read_data_option=(
            ReadDataOption.ReplaceInActiveFrame
            if replace_existing or not frame.has_dataset
            else ReadDataOption.Append
        ),
        reset_style=False,
        variables=FORCE_MOMENT_VARIABLES,
        assign_strand_ids=True,
    )


def add_macro_layout_style(frame=None):
    frame = frame or tp.active_frame()
    plot = frame.plot(PlotType.XYLine)
    plot.activate()
    plot.show_symbols = True

    try:
        tp.macro.execute_command("$!FrameLayout ShowBorder = No")
        tp.macro.execute_command("$!PrintSetup Palette = Color")
        tp.macro.execute_command("$!ExportSetup UseSuperSampleAntiAliasing = Yes")
        tp.macro.execute_command("$!ExportSetup ImageWidth = 2000")
    except TecplotSystemError:
        pass

    x_axis = plot.axes.x_axis(0)
    y_axis = plot.axes.y_axis(0)
    x_axis.log_scale = True

    for axis, offset in ((x_axis, 6), (y_axis, 10)):
        axis.grid_lines.show = True
        axis.minor_grid_lines.show = True
        axis.title.offset = offset

    grid_area = plot.axes.grid_area
    grid_area.show_border = True
    grid_area.border_thickness = 0.1

    legend = plot.legend
    legend.show = True
    legend.box.box_type = TextBox.Filled
    legend.box.margin = 2
    legend.row_spacing = 1.2
    if hasattr(legend, "vertical"):
        legend.vertical = True
    if hasattr(legend, "font"):
        legend.font.size_units = Units.Frame
        legend.font.size = 1.7
    legend.position = (86, 10)
    legend.anchor_alignment = AnchorAlignment.BottomLeft


def add_macro_derived_variables(dataset):
    for equation in DERIVED_VARIABLE_EQUATIONS:
        try:
            tp.data.operate.execute_equation(
                equation,
                ignore_divide_by_zero=True,
            )
        except TecplotSystemError as error:
            print(f"could not evaluate equation: {equation}\n  {error}")

    return dataset


def rename_force_moment_variables(dataset):
    for old_name, display_name in VARIABLE_DISPLAY_NAMES.items():
        try:
            variable = dataset.variable(old_name)
            if variable is not None:
                variable.name = display_name
        except TecplotLogicError:
            continue

    return dataset


def resolve_variable(dataset, name):
    try:
        variable = dataset.variable(name)
        if variable is not None:
            return variable
    except TecplotLogicError:
        pass

    for old_name, display_name in VARIABLE_DISPLAY_NAMES.items():
        if name in (old_name, display_name):
            try:
                variable = dataset.variable(old_name)
                if variable is not None:
                    return variable
            except TecplotLogicError:
                pass

            variable = dataset.variable(display_name)
            if variable is not None:
                return variable

    raise TecplotLogicError(f"could not find variable '{name}'")


def zone_name_matches_alpha(zone_name, alpha):
    match = ALPHA_ZONE_RE.search(zone_name)
    return bool(match and abs(float(match.group(1)) - float(alpha)) < 1.0e-6)


def zone_name_matches_grid_level(zone_name, grid_level):
    match = GRID_ZONE_RE.search(zone_name)
    return bool(match and int(match.group(1)) == int(grid_level))


def zone_values_match(zone, variable, value):
    try:
        values = zone.values(variable)[:]
    except TecplotSystemError:
        return False

    if len(values) == 0:
        return False

    return all(abs(float(item) - float(value)) < 1.0e-6 for item in values)


def selected_zones(
    dataset,
    zone_indices,
    view="alpha",
    alpha=DEFAULT_ALPHA,
    grid_level=1,
):
    zones = [dataset.zone(index) for index in zone_indices]

    if view == "alpha":
        matches = [zone for zone in zones if zone_name_matches_alpha(zone.name, alpha)]
        if matches:
            return matches

        alpha_variable = resolve_variable(dataset, "ALPHA")
        return [zone for zone in zones if zone_values_match(zone, alpha_variable, alpha)]

    if view == "grid":
        matches = [
            zone for zone in zones if zone_name_matches_grid_level(zone.name, grid_level)
        ]
        if matches:
            return matches

        grid_variable = resolve_variable(dataset, "GRID_LEVEL")
        return [zone for zone in zones if zone_values_match(zone, grid_variable, grid_level)]

    return zones


def turbulence_group(metadata):
    model = metadata["aux"].get("TurbulenceModel", "").lower()

    if "sst" in model:
        return "SST_Maps"
    if "wmles" in model:
        return "WMLES_Maps"
    if "qcr" in model and "sa" in model:
        return "SAQ_Maps"
    if "sa" in model:
        return "SA_Maps"

    return "OtherTurbulence_Maps"


def aux_text(metadata, *keys):
    values = []

    for key in keys:
        if metadata["aux"].get(key):
            values.append(metadata["aux"][key])

        values.extend(
            zone["aux"][key]
            for zone in metadata["zones"]
            if zone["aux"].get(key)
        )

    return " ".join(values).lower()


def grid_group(metadata):
    grid_id = aux_text(metadata, "GridId", "GridFileName")

    if "helden" in grid_id:
        return "HeldenUnSt_Maps"
    if "adapt" in grid_id:
        return "CstmUsrAdp_Maps"
    if "cadence" in grid_id and "unstructured" in grid_id:
        return "CadenceUnSTet_Maps"
    if "unstructured" in grid_id:
        return "CstmUsrUns_Maps"
    if "structured" in grid_id or "quad" in grid_id:
        return "CstmUsrStr_Maps"

    return "OtherGrid_Maps"


def _set_aux(container, key, value):
    if key and value is not None:
        container.aux_data[str(key)] = str(value)


def _apply_all_aux_data(target, metadata, zone_metadata=None):
    zone_metadata = zone_metadata or {"aux": {}}
    dataset_aux = metadata.get("aux", {})
    zone_aux = zone_metadata.get("aux", {})

    for key, value in dataset_aux.items():
        _set_aux(target, key, value)

    for key, value in zone_aux.items():
        _set_aux(target, key, value)

    if zone_metadata.get("name"):
        _set_aux(target, "ZoneTitle", zone_metadata["name"])

    _set_aux(target, "TurbulenceGroup", turbulence_group(metadata))
    _set_aux(target, "GridGroup", grid_group(metadata))


def style_linemap(linemap, metadata, style_index, zone_metadata=None):
    color = LINE_COLORS[style_index % len(LINE_COLORS)]
    symbol = SYMBOL_PRESETS[style_index % len(SYMBOL_PRESETS)]

    linemap.line.color = color
    linemap.symbols.show = True
    linemap.symbols.size = 1.0
    linemap.symbols.color = color
    linemap.symbols.fill_mode = FillMode.UseSpecificColor
    linemap.symbols.fill_color = color
    linemap.symbols.symbol_type = SymbolType.Geometry

    geometric_symbol = linemap.symbols.symbol()
    geometric_symbol.shape = symbol

    linemap.aux_data["SourceFile"] = str(metadata["path"])
    linemap.aux_data["TestCase"] = metadata["path"].relative_to(REPO_ROOT).parts[0]
    _apply_all_aux_data(linemap, metadata, zone_metadata=zone_metadata)


def participant_label(metadata):
    aux = metadata["aux"]
    return metadata["title"] or aux.get("ParticipantID") or metadata["path"].parent.name


def create_linemaps_from_loaded_files(
    loaded_files,
    view="alpha",
    alpha=DEFAULT_ALPHA,
    grid_level=1,
    x_variable="GRID_SIZE",
    y_variable="CL_TOT",
):
    frame = tp.active_frame()
    dataset = frame.dataset
    plot = frame.plot(PlotType.XYLine)
    plot.activate()
    plot.delete_linemaps()

    x_var = resolve_variable(dataset, x_variable)
    y_var = resolve_variable(dataset, y_variable)
    linemaps = []

    for style_index, loaded_file in enumerate(loaded_files):
        metadata = loaded_file["metadata"]
        zone_metadata_by_dataset_index = {
            dataset_index: (
                metadata["zones"][zone_offset]
                if zone_offset < len(metadata["zones"])
                else {"name": "", "aux": {}, "numeric_rows": 0}
            )
            for zone_offset, dataset_index in enumerate(loaded_file["zone_indices"])
        }
        zones = selected_zones(
            dataset,
            loaded_file["zone_indices"],
            view=view,
            alpha=alpha,
            grid_level=grid_level,
        )

        for zone_index, zone in enumerate(zones):
            label = participant_label(metadata)
            if len(zones) > 1:
                label = f"{label}.{zone_index + 1}"

            linemap = plot.add_linemap(label, zone=zone, x=x_var, y=y_var)
            linemap.show = True
            _set_linemap_legend_visibility(linemap, show=(zone_index == 0))
            dataset_zone_index = _zone_dataset_index(
                zone,
                dataset,
                candidate_indices=loaded_file["zone_indices"],
            )
            zone_metadata = zone_metadata_by_dataset_index.get(
                dataset_zone_index, {"name": "", "aux": {}, "numeric_rows": 0}
            )
            _apply_all_aux_data(zone, metadata, zone_metadata=zone_metadata)
            _set_aux(zone, "SourceFile", str(metadata["path"]))
            _set_aux(zone, "TestCase", metadata["path"].relative_to(REPO_ROOT).parts[0])
            style_linemap(
                linemap,
                metadata,
                style_index,
                zone_metadata=zone_metadata,
            )
            linemaps.append(linemap)

    return linemaps


def _set_linemap_legend_visibility(linemap, show=True):
    target = LegendShow.Auto if show else LegendShow.Never
    try:
        linemap.show_in_legend = target
        return
    except TecplotSystemError:
        pass

    # Fallback for Tecplot enum/setter inconsistencies across versions.
    try:
        linemap.show_in_legend = 1 if show else 2
    except TecplotSystemError:
        pass


def _zone_dataset_index(zone, dataset, candidate_indices=None):
    zone_index = getattr(zone, "index", None)
    if zone_index is not None:
        return zone_index

    search_indices = (
        candidate_indices if candidate_indices is not None else range(dataset.num_zones)
    )
    for index in search_indices:
        if dataset.zone(index) is zone:
            return index

    raise TecplotLogicError(f"could not resolve dataset index for zone '{zone.name}'")


def load_all(
    test_cases=DEFAULT_TEST_CASES,
    file_pattern=DEFAULT_DATAFILE_PATTERN,
    root=REPO_ROOT,
    view="alpha",
    alpha=DEFAULT_ALPHA,
    grid_level=1,
    x_variable="GRID_SIZE",
    y_variable="CL_TOT",
    reset_layout=True,
):
    if reset_layout:
        tp.new_layout()

    loaded_files = []
    skipped_files = []
    replace_existing_data = True

    for path in find_datafiles(test_cases, file_pattern, root):
        metadata = parse_datafile_metadata(path)
        if metadata["numeric_rows"] == 0:
            reason = "no numeric data rows"
            skipped_files.append((path, reason))
            print(f"SKIPPED: {path}\nREASON: {reason}\n")
            continue

        frame = tp.active_frame()
        zone_count = frame.dataset.num_zones if frame.has_dataset else 0

        try:
            dataset = load_single_datafile(
                path,
                replace_existing=replace_existing_data,
            )
        except TecplotSystemError as error:
            reason = str(error)
            skipped_files.append((path, reason))
            print(f"SKIPPED: {path}\n  {reason}\n")
            continue

        replace_existing_data = False

        loaded_files.append(
            {
                "path": path,
                "metadata": metadata,
                "zone_indices": list(range(zone_count, dataset.num_zones)),
            }
        )

    if not loaded_files:
        print("no data files loaded")
        return []

    dataset = tp.active_frame().dataset
    add_macro_derived_variables(dataset)
    linemaps = create_linemaps_from_loaded_files(
        loaded_files,
        view=view,
        alpha=alpha,
        grid_level=grid_level,
        x_variable=x_variable,
        y_variable=y_variable,
    )
    rename_force_moment_variables(dataset)

    plot = tp.active_frame().plot(PlotType.XYLine)
    plot.view.fit_to_nice()
    add_macro_layout_style()

    return {
        "dataset": dataset,
        "loaded_files": loaded_files,
        "linemaps": linemaps,
        "skipped_files": skipped_files,
    }


def load_test_case(
    test_case,
    file_pattern=DEFAULT_DATAFILE_PATTERN,
    root=REPO_ROOT,
    view="alpha",
    alpha=DEFAULT_ALPHA,
    grid_level=1,
    x_variable="GRID_SIZE",
    y_variable="CL_TOT",
    reset_layout=True,
):
    return load_all(
        test_cases=[test_case],
        file_pattern=file_pattern,
        root=root,
        view=view,
        alpha=alpha,
        grid_level=grid_level,
        x_variable=x_variable,
        y_variable=y_variable,
        reset_layout=reset_layout,
    )


if __name__ == "__main__":
    tp.session.connect()
    load_all()
