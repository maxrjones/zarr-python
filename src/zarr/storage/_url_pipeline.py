"""
Parsing and resolution of URL pipelines (https://github.com/jbms/url-pipeline).

Importing this module is cheap and has no side effects: third-party adapters
(registered through the `zarr.url_adapters` entry-point group) are loaded
only when a pipeline URL naming their scheme is actually resolved.

A string store specification is a URL pipeline only if its root sub-URL
has a URL scheme and every sub-URL after a `|` names a registered adapter
(see `is_url_pipeline`). Every other string, including a local path that
contains `|`, is handled exactly as before.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

from zarr.abc.url_pipeline import (
    AdapterResolution,
    PipelineContext,
    PipelineSegment,
)
from zarr.errors import URLPipelineError
from zarr.registry import get_url_adapter, list_url_adapter_schemes
from zarr.storage._memory import ManagedMemoryStore
from zarr.storage._utils import parse_store_url

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import AccessModeLiteral

__all__ = ["is_url_pipeline", "parse_pipeline", "resolve_pipeline"]

# Adapter scheme per RFC 3986 plus "." to permit vendor-prefixed
# nonstandard schemes (e.g. "earthmover.myscheme").
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*$")

# Root scheme at the very start of the sub-URL. The negative lookahead
# excludes fsspec's chained-URL syntax (``zip::file://...``), which is not a
# URL pipeline and keeps flowing through the fsspec machinery.
_ROOT_SCHEME_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*):(?!:)")

# Schemes zarr resolves natively at the pipeline root. These are never
# dispatched to a registered root adapter, so an installed package cannot
# intercept zarr's own local-path and in-memory routing.
_NATIVE_ROOT_SCHEMES = frozenset({"file", "memory"})

# An absolute Windows drive path (C:\... or C:/...), which counts as an
# absolute path in a `file:` pipeline root.
_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:[/\\]")


def _root_scheme(root_sub_url: str) -> str:
    """
    Detect the scheme of the root sub-URL, or `""` for an opaque root.

    Scheme extraction happens on the raw string (so nothing `urlparse`
    would strip or reject — whitespace, exotic authorities — can desync the
    detected scheme from the body). Single-letter candidates are delegated
    to `parse_store_url`, which knows Windows drive letters are not schemes.
    """
    match = _ROOT_SCHEME_RE.match(root_sub_url)
    if match is None:
        return ""
    scheme = match.group(1)
    if len(scheme) == 1:
        try:
            return parse_store_url(root_sub_url).scheme.lower()
        except ValueError:
            return scheme.lower()
    return scheme.lower()


def _split_query(sub_url: str) -> tuple[str, str | None]:
    """Split a sub-URL on the first `?`. Fragments are not supported."""
    if "#" in sub_url:
        raise URLPipelineError(
            f"URL pipeline sub-URLs do not support fragments: {sub_url!r}. "
            "Percent-encode '#' as '%23' if it is part of the path."
        )
    body, sep, query = sub_url.partition("?")
    return body, query if sep else None


def parse_pipeline(url: str) -> tuple[PipelineSegment, ...]:
    """
    Parse a URL pipeline into its `|`-delimited segments.

    The first segment is the *root* sub-URL; its scheme is detected with the
    same rules as ordinary store URLs (Windows drive letters are not
    schemes). Subsequent segments are *adapter* sub-URLs of the form
    `scheme:body` where the trailing colon is optional when the body is
    empty (`zip` is equivalent to `zip:`).

    A schemeless root (a bare local path) is parsed as opaque text, with no
    query or fragment splitting, since `?` and `#` are ordinary filename
    characters there. Such a string is not a URL pipeline:
    `resolve_pipeline` rejects it and `is_url_pipeline` returns False.

    Segment text is preserved verbatim (no case or percent-encoding
    normalization) except that schemes are lowercased.
    """
    parts = url.split("|")
    if any(not part for part in parts):
        raise URLPipelineError(f"URL pipeline contains an empty sub-URL: {url!r}")

    segments: list[PipelineSegment] = []
    for index, part in enumerate(parts):
        if index == 0:
            scheme = _root_scheme(part)
            if not scheme:
                # opaque root (bare local path, fsspec chained URL, ...)
                segments.append(PipelineSegment(scheme="", body=part, query=None, raw=part))
                continue
            body_and_scheme, query = _split_query(part)
            body = body_and_scheme[len(scheme) + 1 :]
            segments.append(PipelineSegment(scheme=scheme, body=body, query=query, raw=part))
        else:
            body_and_scheme, query = _split_query(part)
            scheme, _, body = body_and_scheme.partition(":")
            scheme = scheme.lower()
            if not _SCHEME_RE.match(scheme):
                raise URLPipelineError(
                    f"invalid adapter scheme {scheme!r} in pipeline segment {part!r}"
                )
            segments.append(PipelineSegment(scheme=scheme, body=body, query=query, raw=part))
    return tuple(segments)


def _root_routes_to_adapter(scheme: str) -> bool:
    """
    Whether a root sub-URL with this scheme is dispatched to a registered
    root adapter. Schemes zarr resolves natively (`file:`, `memory:`) and
    opaque roots are excluded; the registry check inspects entry-point
    names only — no adapter code is imported here.
    """
    return (
        bool(scheme) and scheme not in _NATIVE_ROOT_SCHEMES and scheme in list_url_adapter_schemes()
    )


def _adapter_scheme(sub_url: str) -> str:
    """The lowercased scheme of an adapter sub-URL (`zip` and `zip:a?b` -> `zip`)."""
    return sub_url.partition("?")[0].partition(":")[0].lower()


def is_url_pipeline(url: str) -> bool:
    """
    Whether `url` should be routed through the URL pipeline machinery.

    True when the root sub-URL has a URL scheme and every sub-URL after a `|`
    names a registered adapter (e.g. `s3://bucket/repo|icechunk:`), or when
    there is no `|` and the scheme has a registered *root adapter* (such as
    `gh:`). Anything else — a schemeless local path such as `data|x.zarr`, a
    URL whose `|` is followed by an unregistered scheme — is not a pipeline.
    fsspec chained URLs (`zip::file://...`) and zarr's native `file:` /
    `memory:` schemes are never routed to a root adapter. The registry check
    inspects entry-point names only; no adapter code is imported.
    """
    root, sep, rest = url.partition("|")
    scheme = _root_scheme(root)
    if not sep:
        return _root_routes_to_adapter(scheme)
    if not scheme:
        return False
    registered = list_url_adapter_schemes()
    return all(_adapter_scheme(sub_url) in registered for sub_url in rest.split("|"))


async def resolve_pipeline(
    url: str,
    *,
    mode: AccessModeLiteral | None = None,
    storage_options: dict[str, Any] | None = None,
) -> AdapterResolution:
    """
    Resolve a URL pipeline into a store and a residual path.

    Parameters
    ----------
    url : str
        A URL pipeline, e.g. `"s3://bucket/data.zip|zip:|zarr3:"`.
    mode : AccessModeLiteral | None
        The caller's access mode. `"r"` requires a read-only store; the
        resolver enforces this on whatever the final adapter returns.
    storage_options : dict | None
        Options forwarded to the root sub-URL's store (and visible to
        adapters via the context). Non-dict forms are reserved for future
        per-segment configuration (one mapping per pipeline segment).

    Raises
    ------
    URLPipelineError
        If the URL cannot be parsed, is not a pipeline, names a scheme with
        no registered adapter, names an adapter entry point that fails to
        import, or has a root sub-URL that cannot be resolved.
    TypeError
        If `storage_options` are passed to a root that does not accept them
        (local paths and `memory:`), as for non-pipeline stores.
    OSError
        Errors from opening the root resource (e.g. a missing local directory
        in mode `"r"`) propagate unchanged, as for non-pipeline stores.
    """
    segments = parse_pipeline(url)
    if not segments[0].scheme:
        raise URLPipelineError(
            f"{url!r} is not a URL pipeline: its root sub-URL has no scheme; "
            "spell a local root as an absolute 'file:' URL"
        )
    if len(segments) == 1 and not _root_routes_to_adapter(segments[0].scheme):
        raise URLPipelineError(
            f"{url!r} is not a URL pipeline: it has no '|' separator and no "
            f"URL pipeline adapter is registered for scheme {segments[0].scheme!r}"
        )
    return await _resolve(segments, mode=mode, storage_options=storage_options)


async def _resolve(
    segments: tuple[PipelineSegment, ...],
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> AdapterResolution:
    if len(segments) == 1 and not _root_routes_to_adapter(segments[0].scheme):
        return AdapterResolution(
            store=await _resolve_root(segments[0], mode=mode, storage_options=storage_options)
        )

    *preceding, last = segments
    adapter_cls = get_url_adapter(last.scheme)

    context = PipelineContext(
        preceding=tuple(preceding),
        mode=mode,
        storage_options=storage_options,
    )
    resolution = await adapter_cls.open_pipeline_segment(last, context)
    if mode == "r" and not resolution.store.read_only:
        # The caller required read-only; enforce it rather than trusting
        # the adapter to have honored context.read_only.
        try:
            read_only_store = resolution.store.with_read_only(True)
            await read_only_store._ensure_open()
        except NotImplementedError as exc:
            resolution.store.close()
            raise URLPipelineError(
                f"adapter {last.scheme!r} returned a writable store for mode 'r', "
                "and the store does not support read-only conversion via "
                ".with_read_only()"
            ) from exc
        except Exception:
            resolution.store.close()
            raise
        resolution = dataclasses.replace(resolution, store=read_only_store)
    return resolution


async def _resolve_root(
    segment: PipelineSegment,
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> Store:
    """
    Resolve the root sub-URL of a pipeline into a store.

    `memory:` and `file:` roots are resolved here with the URL pipeline
    spec's semantics (spelling equivalences, mandatory absolute `file:`
    paths); everything else (e.g. fsspec URLs) delegates to
    the existing `StoreLike` machinery unchanged.
    """
    if segment.scheme == "memory":
        return _resolve_memory_root(segment, mode=mode, storage_options=storage_options)
    from zarr.storage._common import make_store  # circular import

    if segment.scheme == "file":
        # Per the spec: file://localhost/p and file:///p are equivalent to
        # file:/p; other authorities are unsupported; relative paths are
        # forbidden; file: URLs carry no query.
        if segment.query is not None:
            raise URLPipelineError(f"'file:' pipeline roots do not accept a query: {segment.raw!r}")
        body = segment.body
        if body.startswith("//"):
            authority, sep, rest = body[2:].partition("/")
            if authority not in ("", "localhost"):
                raise URLPipelineError(
                    f"unsupported authority {authority!r} in 'file:' pipeline "
                    f"root {segment.raw!r}; only an empty authority or "
                    "'localhost' is allowed"
                )
            body = f"/{rest}" if sep else ""
        # a file URL spells a Windows drive path as /C:/...; strip the
        # leading slash so the local-path machinery sees the drive
        if body.startswith("/") and _WINDOWS_DRIVE_RE.match(body[1:]):
            body = body[1:]
        if not (body.startswith("/") or _WINDOWS_DRIVE_RE.match(body)):
            raise URLPipelineError(
                f"'file:' pipeline roots must carry an absolute path: {segment.raw!r}"
            )
        return await make_store(f"file:{body}", mode=mode, storage_options=storage_options)

    try:
        return await make_store(segment.raw, mode=mode, storage_options=storage_options)
    except ValueError as exc:
        raise URLPipelineError(
            f"could not resolve the pipeline root {segment.raw!r}: {exc}"
        ) from exc


def _resolve_memory_root(
    segment: PipelineSegment,
    *,
    mode: AccessModeLiteral | None,
    storage_options: dict[str, Any] | None,
) -> Store:
    """
    Resolve a `memory:` pipeline root per the spec's spelling equivalences:
    `memory:` ≡ `memory:/` ≡ `memory://`, and `memory:a` ≡ `memory:/a` ≡
    `memory://a`. The first path component names the managed store; the
    remainder is a path within it.
    """
    if segment.query is not None:
        raise URLPipelineError(f"'memory:' pipeline roots do not accept a query: {segment.raw!r}")
    if storage_options:
        raise TypeError(
            "'storage_options' was provided but unused. "
            "'storage_options' is only used when the store is passed as an FSSpec URI string.",
        )
    name, _, path = segment.body.lstrip("/").partition("/")
    return ManagedMemoryStore(name=name, path=path, read_only=mode == "r")
