import tecplot as tp
from tecplot.constant import AnchorAlignment, GeomShape, PlotType, TextBox, TileMode, Units
from tecplot.exception import TecplotLogicError, TecplotSystemError

try:
    import load_data
    from load_data import LINE_COLORS
except ImportError:
    from python_module import load_data
    from python_module.load_data import LINE_COLORS


def _active_xy_plot():
    frame = tp.active_frame()
    if frame is None:
        raise TecplotLogicError("no active frame")
    return frame.plot(PlotType.XYLine)


def _enforce_legend_style(plot):
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


def _iter_linemaps(plot):
    if hasattr(plot, "linemaps"):
        linemaps_obj = plot.linemaps
        if callable(linemaps_obj):
            linemaps_obj = linemaps_obj()
        try:
            for linemap in linemaps_obj:
                yield linemap
            return
        except TypeError:
            pass

    if not hasattr(plot, "num_linemaps") or not hasattr(plot, "linemap"):
        raise TecplotLogicError("could not access linemaps from active XY plot")

    for index in range(plot.num_linemaps):
        yield plot.linemap(index)


def _iter_frames(page):
    if hasattr(page, "frames"):
        frames_obj = page.frames
        if callable(frames_obj):
            frames_obj = frames_obj()
        try:
            for frame in frames_obj:
                yield frame
            return
        except TypeError:
            pass

    if not hasattr(page, "num_frames") or not hasattr(page, "frame"):
        raise TecplotLogicError("could not access frames from active page")

    for index in range(page.num_frames):
        yield page.frame(index)


def _iter_dataset_zones(dataset):
    if hasattr(dataset, "zones"):
        zones_obj = dataset.zones
        if callable(zones_obj):
            zones_obj = zones_obj()
        try:
            for zone in zones_obj:
                yield zone
            return
        except TypeError:
            pass

    if not hasattr(dataset, "num_zones") or not hasattr(dataset, "zone"):
        raise TecplotLogicError("could not access zones from dataset")

    for index in range(dataset.num_zones):
        yield dataset.zone(index)


def _iter_dataset_variables(dataset):
    if hasattr(dataset, "variables"):
        variables_obj = dataset.variables
        if callable(variables_obj):
            variables_obj = variables_obj()
        try:
            for variable in variables_obj:
                yield variable
            return
        except TypeError:
            pass

    if not hasattr(dataset, "num_variables") or not hasattr(dataset, "variable"):
        raise TecplotLogicError("could not access variables from dataset")

    for index in range(dataset.num_variables):
        yield dataset.variable(index)


def _delete_zones(dataset, zones):
    if not zones:
        return

    if hasattr(dataset, "delete_zones"):
        try:
            dataset.delete_zones(zones)
            return
        except TypeError:
            pass
        try:
            dataset.delete_zones([zone.index for zone in zones])
            return
        except Exception:
            pass

    for zone in zones:
        if hasattr(zone, "delete"):
            zone.delete()
            continue
        raise TecplotLogicError("could not delete one or more zones")


def _delete_variables(dataset, variables):
    if not variables:
        return

    if hasattr(dataset, "delete_variables"):
        try:
            dataset.delete_variables(variables)
            return
        except TypeError:
            pass
        try:
            dataset.delete_variables([variable.index for variable in variables])
            return
        except Exception:
            pass

    for variable in variables:
        if hasattr(variable, "delete"):
            variable.delete()
            continue
        raise TecplotLogicError("could not delete one or more variables")


def _aux_matches_text(aux_data, match_text, case_sensitive=False):
    if not match_text:
        return True

    needle = str(match_text) if case_sensitive else str(match_text).lower()
    for key, value in aux_data.items():
        haystack = f"{key}={value}"
        haystack = haystack if case_sensitive else haystack.lower()
        if needle in haystack:
            return True

    return False


def _next_unused_line_color(plot):
    used_colors = set()
    for linemap in _iter_linemaps(plot):
        if linemap.show:
            used_colors.add(linemap.line.color)

    for color in LINE_COLORS:
        if color not in used_colors:
            return color

    return LINE_COLORS[0]


def enable_linemaps_with_zone_aux(
    match_text,
    case_sensitive=False,
    disable_other_linemaps=False,
):
    plot = _active_xy_plot()
    for linemap in _iter_linemaps(plot):
        zone = linemap.zone
        matches = _aux_matches_text(zone.aux_data, match_text, case_sensitive)
        visible = matches or not disable_other_linemaps
        linemap.show = visible
        try:
            zone.active = visible
        except (AttributeError, TecplotSystemError):
            pass
    _enforce_legend_style(plot)


def color_linemaps_by_zone_aux(aux_text):
    plot = _active_xy_plot()
    for linemap in _iter_linemaps(plot):
        zone = linemap.zone
        visible = _aux_matches_text(zone.aux_data, aux_text, case_sensitive=False)
        linemap.show = visible
        try:
            zone.active = visible
        except (AttributeError, TecplotSystemError):
            pass
    _enforce_legend_style(plot)


def enable_linemaps_for_zone(
    *,
    zones_containing_string,
    disable_other_zones=True,
    same_color=True,
):
    target = str(zones_containing_string).strip().lower()
    plot = _active_xy_plot()
    shared_color = None
    active_linemaps = []
    for linemap in _iter_linemaps(plot):
        zone = linemap.zone
        zone_text = zone.name.strip().lower()
        matches = target in zone_text
        visible = matches or not disable_other_zones
        linemap.show = visible
        if visible:
            active_linemaps.append(linemap)
        try:
            zone.active = visible
        except (AttributeError, TecplotSystemError):
            pass

    if same_color and active_linemaps:
        shared_color = _next_unused_line_color(plot)
        for linemap in active_linemaps:
            linemap.line.color = shared_color
            linemap.symbols.color = shared_color
            linemap.symbols.fill_color = shared_color

    _enforce_legend_style(plot)


def enable_linemap(linemap_name, disable_other_linemaps=False):
    target = str(linemap_name).strip().lower()
    plot = _active_xy_plot()
    for linemap in _iter_linemaps(plot):
        matches = linemap.name.strip().lower() == target
        visible = matches or not disable_other_linemaps
        linemap.show = visible
        try:
            linemap.zone.active = visible
        except (AttributeError, TecplotSystemError):
            pass
    _enforce_legend_style(plot)


def enable_all_linemaps():
    plot = _active_xy_plot()
    for linemap in _iter_linemaps(plot):
        linemap.show = True
        try:
            linemap.zone.active = True
        except (AttributeError, TecplotSystemError):
            pass
    _enforce_legend_style(plot)


def set_linemaps_axis_variable(*, axis="y", variable):
    axis_name = str(axis).strip().lower()
    if axis_name not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    variable_name = str(variable).strip()
    if not variable_name:
        raise ValueError("variable is required")

    frame = tp.active_frame()
    if frame is None:
        raise TecplotLogicError("no active frame")
    dataset = frame.dataset

    resolved_variable = load_data.resolve_variable(dataset, variable_name)
    variable_index = resolved_variable.index

    plot = _active_xy_plot()
    for linemap in _iter_linemaps(plot):
        if axis_name == "x":
            linemap.x_variable_index = variable_index
        else:
            linemap.y_variable_index = variable_index


def _set_nested_attr(obj, attr_path, value):
    parts = str(attr_path).split(".")
    if not parts:
        raise ValueError("attribute_path is required")

    if str(attr_path) == "symbols.shape":
        symbol = obj.symbols.symbol()
        if isinstance(value, str):
            shape_map = {
                "square": GeomShape.Square,
                "triangle": GeomShape.Del,
                "delta": GeomShape.Del,
                "gradient": GeomShape.Del,
                "circle": GeomShape.Circle,
                "diamond": GeomShape.Diamond,
            }
            key = value.strip().lower()
            if key not in shape_map:
                raise ValueError(f"unsupported symbols.shape value '{value}'")
            symbol.shape = shape_map[key]
        else:
            symbol.shape = value
        return

    target = obj
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise AttributeError(f"'{type(target).__name__}' has no attribute '{part}'")
        target = getattr(target, part)

    last = parts[-1]
    if not hasattr(target, last):
        raise AttributeError(f"'{type(target).__name__}' has no attribute '{last}'")
    setattr(target, last, value)


def set_linemap_attribute_by_aux(
    *,
    aux_key,
    aux_value,
    attribute_path,
    value,
    aux_source="zone",
    match_mode="contains",
    case_sensitive=False,
):
    key_text = str(aux_key).strip()
    if not key_text:
        raise ValueError("aux_key is required")

    value_text = str(aux_value).strip()
    if not value_text:
        raise ValueError("aux_value is required")

    source_name = str(aux_source).strip().lower()
    if source_name not in ("zone", "linemap"):
        raise ValueError("aux_source must be 'zone' or 'linemap'")

    mode = str(match_mode).strip().lower()
    if mode not in ("contains", "equals"):
        raise ValueError("match_mode must be 'contains' or 'equals'")

    value_cmp = value_text if case_sensitive else value_text.lower()

    plot = _active_xy_plot()
    updated_count = 0
    for linemap in _iter_linemaps(plot):
        aux_data = linemap.zone.aux_data if source_name == "zone" else linemap.aux_data
        raw = None
        for aux_key, aux_entry_value in aux_data.items():
            k = str(aux_key) if case_sensitive else str(aux_key).lower()
            wanted_key = key_text if case_sensitive else key_text.lower()
            if k == wanted_key:
                raw = aux_entry_value
                break
        if raw is None:
            continue

        raw_text = str(raw)
        raw_cmp = raw_text if case_sensitive else raw_text.lower()
        key_ok = True
        if mode == "equals":
            value_ok = raw_cmp == value_cmp
        else:
            value_ok = value_cmp in raw_cmp

        if key_ok and value_ok:
            _set_nested_attr(linemap, attribute_path, value)
            updated_count += 1

    return updated_count


def enable_linemaps_by_aux(
    *,
    aux_key,
    aux_value,
    aux_source="zone",
    match_mode="contains",
    case_sensitive=False,
    disable_other_linemaps=True,
):
    key_text = str(aux_key).strip()
    if not key_text:
        raise ValueError("aux_key is required")

    value_text = str(aux_value).strip()
    if not value_text:
        raise ValueError("aux_value is required")

    source_name = str(aux_source).strip().lower()
    if source_name not in ("zone", "linemap"):
        raise ValueError("aux_source must be 'zone' or 'linemap'")

    mode = str(match_mode).strip().lower()
    if mode not in ("contains", "equals"):
        raise ValueError("match_mode must be 'contains' or 'equals'")

    wanted_key = key_text if case_sensitive else key_text.lower()
    wanted_value = value_text if case_sensitive else value_text.lower()

    plot = _active_xy_plot()
    enabled_count = 0
    for linemap in _iter_linemaps(plot):
        aux_data = linemap.zone.aux_data if source_name == "zone" else linemap.aux_data
        raw = None
        for aux_key_name, aux_entry_value in aux_data.items():
            key_cmp = str(aux_key_name) if case_sensitive else str(aux_key_name).lower()
            if key_cmp == wanted_key:
                raw = aux_entry_value
                break

        matches = False
        if raw is not None:
            raw_text = str(raw)
            raw_cmp = raw_text if case_sensitive else raw_text.lower()
            if mode == "equals":
                matches = raw_cmp == wanted_value
            else:
                matches = wanted_value in raw_cmp

        visible = matches or not disable_other_linemaps
        linemap.show = visible
        try:
            linemap.zone.active = visible
        except (AttributeError, TecplotSystemError):
            pass
        if matches:
            enabled_count += 1

    _enforce_legend_style(plot)
    return enabled_count


def reset_view():
    active_frame = tp.active_frame()
    if active_frame is None:
        raise TecplotLogicError("no active frame")

    page = active_frame.page
    frames = list(_iter_frames(page))
    if not frames:
        raise TecplotLogicError("no frames found on active page")

    original_frame = frames[0]
    for frame in reversed(frames[1:]):
        if hasattr(page, "delete_frame"):
            page.delete_frame(frame)
        elif hasattr(frame, "delete"):
            frame.delete()
        else:
            raise TecplotLogicError("could not delete frame from active page")

    dataset = original_frame.dataset
    trimmed_zones = [
        zone for zone in _iter_dataset_zones(dataset) if zone.name.startswith("trimmed_")
    ]
    _delete_zones(dataset, trimmed_zones)

    diff_variables = [
        variable
        for variable in _iter_dataset_variables(dataset)
        if str(variable.name).endswith(" diff")
    ]
    _delete_variables(dataset, diff_variables)

    with original_frame.activated():
        enable_all_linemaps()
        plot = _active_xy_plot()
        plot.activate()
        plot.view.fit_to_nice()

    page.tile_frames(TileMode.Columns)
