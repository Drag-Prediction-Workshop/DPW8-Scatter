import bisect
import math

import tecplot as tp
from tecplot.constant import PlotType, TileMode
from tecplot.exception import TecplotLogicError, TecplotSystemError
from tecplot.tecutil import _tecutil

def _to_sorted_xy(zone, xvar, yvar):
    x_values = [float(v) for v in zone.values(xvar)[:]]
    y_values = [float(v) for v in zone.values(yvar)[:]]
    points = sorted(zip(x_values, y_values), key=lambda pair: pair[0])
    if not points:
        return [], []
    x_sorted = [p[0] for p in points]
    y_sorted = [p[1] for p in points]
    return x_sorted, y_sorted


def _unique_sorted(values):
    seen = set()
    out = []
    for value in sorted(values):
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _interp_linear(x_new, x_old, y_old):
    if not x_old:
        raise ValueError("cannot interpolate with empty source data")
    if len(x_old) == 1:
        return [y_old[0] for _ in x_new]

    y_new = []
    for xn in x_new:
        if xn <= x_old[0]:
            y_new.append(y_old[0])
            continue
        if xn >= x_old[-1]:
            y_new.append(y_old[-1])
            continue

        right = bisect.bisect_left(x_old, xn)
        left = max(0, right - 1)
        x0, x1 = x_old[left], x_old[right]
        y0, y1 = y_old[left], y_old[right]
        if math.isclose(x1, x0):
            y_new.append(y0)
            continue

        t = (xn - x0) / (x1 - x0)
        y_new.append(y0 + t * (y1 - y0))

    return y_new

def trim_linemaps_to_common_domain(lmap1, lmap2):
    dataset = tp.active_frame().dataset
    plot = tp.active_frame().plot(PlotType.XYLine)
    zone1 = dataset.zone(lmap1.zone_index)
    zone2 = dataset.zone(lmap2.zone_index)

    xvar1 = dataset.variable(lmap1.x_variable_index)
    yvar1 = dataset.variable(lmap1.y_variable_index)
    xvar2 = dataset.variable(lmap2.x_variable_index)
    yvar2 = dataset.variable(lmap2.y_variable_index)

    x1, y1 = _to_sorted_xy(zone1, xvar1, yvar1)
    x2, y2 = _to_sorted_xy(zone2, xvar2, yvar2)

    if not x1 or not x2:
        return None, None

    xmin = max(x1[0], x2[0])
    xmax = min(x1[-1], x2[-1])
    if xmin >= xmax:
        return None, None

    x_new = _unique_sorted(
        [xmin]
        + [x for x in x1 if xmin < x < xmax]
        + [x for x in x2 if xmin < x < xmax]
        + [xmax]
    )

    y1_new = _interp_linear(x_new, x1, y1)
    y2_new = _interp_linear(x_new, x2, y2)
    z1_new = dataset.add_ordered_zone("trimmed_" + lmap1.name, (len(x_new),))
    z2_new = dataset.add_ordered_zone("trimmed_" + lmap2.name, (len(x_new),))

    z1_new.values(xvar1)[:] = x_new
    z1_new.values(yvar1)[:] = y1_new
    z2_new.values(xvar2)[:] = x_new
    z2_new.values(yvar2)[:] = y2_new

    lm1_new = plot.add_linemap()
    lm1_new.name = "trimmed_" + lmap1.name
    lm1_new.zone_index = z1_new.index
    lm1_new.x_variable_index = xvar1.index
    lm1_new.y_variable_index = yvar1.index
    lm1_new.show_in_legend = lmap1.show_in_legend

    lm2_new = plot.add_linemap()
    lm2_new.name = "trimmed_" + lmap2.name
    lm2_new.zone_index = z2_new.index
    lm2_new.x_variable_index = xvar2.index
    lm2_new.y_variable_index = yvar2.index
    lm2_new.show_in_legend = lmap2.show_in_legend

    lm1_new.show = True
    lm2_new.show = True
    lmap1.show = False
    lmap2.show = False
    return lm1_new, lm2_new


def create_y_difference_variable(lmap1, lmap2, zone_name=None):
    dataset = tp.active_frame().dataset
    zone1 = dataset.zone(lmap1.zone_index)
    zone2 = dataset.zone(lmap2.zone_index)
    yvar1 = dataset.variable(lmap1.y_variable_index)
    yvar2 = dataset.variable(lmap2.y_variable_index)

    zone_label = str(zone_name).strip() if zone_name is not None else ""
    zone_label = zone_label.replace("{", "(").replace("}", ")")
    variable_name = "diff"
    equation = (
        f"{{{variable_name}}} = "
        f"{{{yvar1.name}}}[{zone1.index + 1}] - "
        f"{{{yvar2.name}}}[{zone2.index + 1}]"
    )
    tp.data.operate.execute_equation(equation, zones=[zone1])
    return dataset.variable(variable_name)


def fit_view(*frames):
    if not frames:
        frames = (tp.active_frame(),)

    for frame in frames:
        with frame.activated():
            plot = frame.plot(PlotType.XYLine)
            plot.activate()
            plot.view.fit_to_nice()


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


def _copy_xy_axis_style(source_axis, target_axis):
    for attr in ("log_scale", "reverse", "min", "max"):
        if hasattr(source_axis, attr) and hasattr(target_axis, attr):
            try:
                setattr(target_axis, attr, getattr(source_axis, attr))
            except Exception:
                pass

    for attr in ("grid_lines", "minor_grid_lines", "title"):
        if not hasattr(source_axis, attr) or not hasattr(target_axis, attr):
            continue
        source_obj = getattr(source_axis, attr)
        target_obj = getattr(target_axis, attr)
        for sub_attr in ("show", "offset", "text"):
            if hasattr(source_obj, sub_attr) and hasattr(target_obj, sub_attr):
                try:
                    setattr(target_obj, sub_attr, getattr(source_obj, sub_attr))
                except Exception:
                    pass


def _copy_xy_plot_style(source_plot, target_plot):
    if hasattr(source_plot, "show_symbols") and hasattr(target_plot, "show_symbols"):
        try:
            target_plot.show_symbols = source_plot.show_symbols
        except Exception:
            pass

    try:
        _copy_xy_axis_style(source_plot.axes.x_axis(0), target_plot.axes.x_axis(0))
        _copy_xy_axis_style(source_plot.axes.y_axis(0), target_plot.axes.y_axis(0))
    except Exception:
        pass

    if hasattr(source_plot, "legend") and hasattr(target_plot, "legend"):
        src_legend = source_plot.legend
        dst_legend = target_plot.legend
        for attr in ("show", "position", "anchor_alignment", "row_spacing"):
            if hasattr(src_legend, attr) and hasattr(dst_legend, attr):
                try:
                    setattr(dst_legend, attr, getattr(src_legend, attr))
                except Exception:
                    pass
        if hasattr(src_legend, "box") and hasattr(dst_legend, "box"):
            for attr in ("box_type", "margin"):
                if hasattr(src_legend.box, attr) and hasattr(dst_legend.box, attr):
                    try:
                        setattr(dst_legend.box, attr, getattr(src_legend.box, attr))
                    except Exception:
                        pass
        if hasattr(src_legend, "font") and hasattr(dst_legend, "font"):
            for attr in ("size", "size_units"):
                if hasattr(src_legend.font, attr) and hasattr(dst_legend.font, attr):
                    try:
                        setattr(dst_legend.font, attr, getattr(src_legend.font, attr))
                    except Exception:
                        pass


def _copy_linemap_style(source_linemap, target_linemap):
    if hasattr(source_linemap, "line") and hasattr(target_linemap, "line"):
        for attr in ("color", "line_thickness", "pattern"):
            if hasattr(source_linemap.line, attr) and hasattr(target_linemap.line, attr):
                try:
                    setattr(target_linemap.line, attr, getattr(source_linemap.line, attr))
                except Exception:
                    pass

    if hasattr(source_linemap, "symbols") and hasattr(target_linemap, "symbols"):
        for attr in ("show", "size", "color", "fill_color", "fill_mode", "symbol_type"):
            if hasattr(source_linemap.symbols, attr) and hasattr(target_linemap.symbols, attr):
                try:
                    setattr(
                        target_linemap.symbols,
                        attr,
                        getattr(source_linemap.symbols, attr),
                    )
                except Exception:
                    pass
        try:
            src_sym = source_linemap.symbols.symbol()
            dst_sym = target_linemap.symbols.symbol()
            if hasattr(src_sym, "shape") and hasattr(dst_sym, "shape"):
                dst_sym.shape = src_sym.shape
        except Exception:
            pass

    if hasattr(source_linemap, "show_in_legend") and hasattr(target_linemap, "show_in_legend"):
        try:
            target_linemap.show_in_legend = source_linemap.show_in_legend
        except Exception:
            pass


def plot_linemap_diff(lmap1, lmap2):
    source_frame = lmap1.plot.frame
    if source_frame != lmap2.plot.frame:
        raise ValueError("linemaps must belong to the same frame")
    zone_label = lmap1.zone.name

    with source_frame.activated():
        trimmed_lmap1, trimmed_lmap2 = trim_linemaps_to_common_domain(lmap1, lmap2)
        if trimmed_lmap1 is None or trimmed_lmap2 is None:
            return None
        diff_var = create_y_difference_variable(
            trimmed_lmap1,
            trimmed_lmap2,
            zone_name=zone_label,
        )
        trimmed_lmap1.show = False
        trimmed_lmap2.show = False
        lmap1.show = True
        lmap2.show = True

    dataset = source_frame.dataset
    page = source_frame.page
    diff_frame = page.add_frame()
    if not _tecutil.FrameSetDataSet(dataset.uid, diff_frame.uid):
        raise TecplotSystemError("could not attach dataset to new frame")

    with diff_frame.activated():
        diff_dataset = diff_frame.dataset
        diff_plot = diff_frame.plot(PlotType.XYLine)
        diff_plot.activate()
        diff_plot.delete_linemaps()
        _copy_xy_plot_style(source_frame.plot(PlotType.XYLine), diff_plot)

        diff_linemap = diff_plot.add_linemap(
            diff_var.name,
            zone=diff_dataset.zone(trimmed_lmap1.zone_index),
            x=diff_dataset.variable(trimmed_lmap1.x_variable_index),
            y=diff_dataset.variable(diff_var.index),
        )
        _copy_linemap_style(lmap1, diff_linemap)
        diff_linemap.show = True
        try:
            diff_plot.axes.y_axis(0).title.text = diff_var.name
        except Exception:
            pass

    page.tile_frames(TileMode.Columns)

    for frame in (source_frame, diff_frame):
        with frame.activated():
            plot = frame.plot(PlotType.XYLine)
            plot.activate()
            linking = plot.linking_between_frames
            linking.group = 1
            linking.link_x_axis_range = True

    fit_view(diff_frame, source_frame)
    return diff_linemap


def plot_active_linemaps_diff():
    frame = tp.active_frame()
    if frame is None:
        raise TecplotLogicError("no active frame")

    plot = frame.plot(PlotType.XYLine)
    active_linemaps = [linemap for linemap in _iter_linemaps(plot) if linemap.show]
    if len(active_linemaps) != 2:
        raise ValueError(
            f"expected exactly 2 active linemaps, found {len(active_linemaps)}"
        )

    return plot_linemap_diff(active_linemaps[0], active_linemaps[1])
