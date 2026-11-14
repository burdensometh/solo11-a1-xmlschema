#
# Copyright (c), 2016-2026, SISSA (International School for Advanced Studies).
# All rights reserved.
# This file is distributed under the terms of the MIT License.
# See the file 'LICENSE' in the root directory of the present
# distribution, or http://opensource.org/licenses/MIT.
#
# @author Davide Brunato <brunato@sissa.it>
#
"""
Helpers for tracking the version of the XSD sources a schema has been built from,
for detecting stale schema instances and for reloading schemas atomically.

A *fingerprint* is a content-based identifier of a single XSD resource: local and
remote resources are read and hashed, in-memory sources are hashed from their
serialized tree. Fingerprints don't depend on HTTP caches, file handles or process
lifetime, so two processes (or the same process before and after a restart) can
reliably tell whether they are using the same XSD rules.

A :class:`SchemaRegistry` keeps the single canonical schema instance for each
location and swaps it atomically on reload, so cached references can never serve
a half-built or an older set of rules.
"""
import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from xmlschema.aliases import SchemaType, SourceArgType
from xmlschema.exceptions import XMLSchemaReloadError, XMLResourceOSError
from xmlschema.resources import XMLResource
from xmlschema.utils.urls import get_url, is_local_url, normalize_url

if TYPE_CHECKING:
    from xmlschema.settings import SchemaSettings

__all__ = (
    'SourceFingerprint', 'SchemaFingerprint', 'fingerprint_resource',
    'resource_is_stale', 'fingerprint_schemas', 'SchemaRegistry', 'registry',
)

#: Number of hexadecimal characters exposed by the short fingerprint.
FINGERPRINT_HEX_SIZE = 12


@dataclass(frozen=True)
class SourceFingerprint:
    """
    Fingerprint of a single XSD resource.

    :ivar url: the normalized URL of the resource, also set for in-memory sources \
    when one is derivable, otherwise ``None``.
    :ivar digest: SHA-256 hexadecimal digest of the resource content.
    :ivar size: size in bytes of the hashed content.
    :ivar mtime: modification time of a local file resource, otherwise ``None``.
    """
    url: Optional[str]
    digest: str
    size: int
    mtime: Optional[float] = None

    @property
    def short(self) -> str:
        """A shortened digest, suitable for logs and version strings."""
        return self.digest[:FINGERPRINT_HEX_SIZE]

    def __str__(self) -> str:
        return f'{self.short}:{self.url or "<in-memory>"}'


@dataclass(frozen=True)
class SchemaFingerprint:
    """
    Fingerprint of all the XSD resources (main schema plus includes and imports)
    that compose a built schema, ordered by URL so that the same set of sources
    always produces the same identifier.

    :ivar sources: fingerprints of the composing XSD resources.
    :ivar digest: SHA-256 hexadecimal digest computed over the resource digests.
    """
    sources: tuple[SourceFingerprint, ...]
    digest: str

    @property
    def short(self) -> str:
        """A shortened digest, suitable for logs and version strings."""
        return self.digest[:FINGERPRINT_HEX_SIZE]

    def __str__(self) -> str:
        return self.short


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_resource(resource: XMLResource) -> SourceFingerprint:
    """
    Computes the fingerprint of an XSD resource, reading its current content.

    :raises XMLResourceOSError: if the resource URL is not accessible.
    """
    url = resource.url
    mtime: Optional[float] = None

    if url is not None and is_local_url(url) and (filepath := resource.filepath):
        # stat() is used only to record a cheap change hint; the digest below
        # is always computed from the actual content.
        try:
            mtime = os.stat(filepath).st_mtime
        except OSError:
            mtime = None

    if resource.text is not None:
        data = resource.text.encode('utf-8')
    else:
        probe = resource.__class__(
            resource._source,
            base_url=resource.base_url,
            allow=resource.allow,
            defuse=resource.defuse,
            timeout=resource.timeout,
            block=resource.block,
            uri_mapper=resource.uri_mapper,
            opener=resource.opener,
        )
        probe.load()
        if probe.text is None:
            # Element/ElementTree source: serialize the tree
            probe.text = probe.tostring(xml_declaration=True)
        data = probe.text.encode('utf-8')

    return SourceFingerprint(url=url, digest=_hash_bytes(data), size=len(data), mtime=mtime)


def resource_is_stale(resource: XMLResource,
                      fingerprint: SourceFingerprint,
                      check_content: bool = False) -> bool:
    """
    Tells whether an XSD resource changed since *fingerprint* was computed.

    :param resource: the resource to check.
    :param fingerprint: the fingerprint recorded when the resource was last loaded.
    :param check_content: for default a local file change is detected with a cheap
        ``stat()`` comparison (size and modification time). Set this to ``True`` to
        always re-read and hash the content, which is required for remote resources
        and for file systems with coarse or unreliable modification times.
    :raises XMLResourceOSError: if the resource URL is not accessible anymore.
    """
    url = resource.url
    if not check_content and url is not None and is_local_url(url) \
            and fingerprint.mtime is not None and (filepath := resource.filepath):
        try:
            stat = os.stat(filepath)
        except OSError as err:
            raise XMLResourceOSError(err) from err
        return stat.st_mtime != fingerprint.mtime or stat.st_size != fingerprint.size

    return fingerprint_resource(resource).digest != fingerprint.digest


def _is_library_schema(schema: SchemaType) -> bool:
    """Tells whether a schema ships with the library (meta-schema and bases)."""
    url = schema.source.url or ''
    return '/xmlschema/schemas/' in url or schema.is_meta()


def fingerprint_schemas(schema: SchemaType) -> SchemaFingerprint:
    """
    Computes the fingerprint of every XSD resource that composes *schema*
    (the main schema and its included/imported schemas). Predefined schemas
    shipped with the library are excluded: they change only with a library
    upgrade and are not part of the user's XSD rules.
    """
    fingerprints = sorted(
        (
            fingerprint_resource(s.source)
            for s in schema.maps.iter_schemas()
            if not _is_library_schema(s)
        ),
        key=lambda x: x.url or '',
    )
    payload = ''.join(f'{f.url}\0{f.digest}\n' for f in fingerprints).encode('utf-8')
    return SchemaFingerprint(sources=tuple(fingerprints), digest=_hash_bytes(payload))


class SchemaRegistry:
    """
    A thread-safe registry that keeps one canonical :class:`XMLSchema` instance
    per location. All consumers obtain the same instance through :meth:`get`, so a
    :meth:`reload` atomically swaps the rules used by every subsequent request,
    including references previously cached by callers.

    :param schema_class: the schema class to instantiate (default is
        :class:`~xmlschema.XMLSchema10`).
    :param settings: optional :class:`~xmlschema.settings.SchemaSettings` applied to
        every schema created by the registry.
    """

    def __init__(self,
                 schema_class: Optional[type[SchemaType]] = None,
                 settings: Optional['SchemaSettings'] = None) -> None:
        self.schema_class = schema_class
        self.settings = settings
        self._lock = threading.RLock()
        self._schemas: dict[str, SchemaType] = {}
        self._fingerprints: dict[str, SchemaFingerprint] = {}

    def __contains__(self, location: object) -> bool:
        return location in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)

    def __iter__(self):
        with self._lock:
            yield from list(self._schemas)

    @staticmethod
    def _key_of(schema: SchemaType) -> str:
        source = schema.source
        if source.url is not None:
            return source.url
        return f'id:{id(source)}'

    def register(self, schema: SchemaType,
                 fingerprint: Optional[SchemaFingerprint] = None) -> str:
        """
        Registers an already built schema instance under its source location.
        If a schema is already registered at the same location it is replaced
        atomically (the previous instance is updated in place). The fingerprint is
        computed when not provided. Returns the registry key of the schema.
        """
        key = self._key_of(schema)
        with self._lock:
            previous = self._schemas.get(key)
            if previous is not None and previous is not schema:
                previous.reload_from(schema)
            else:
                self._schemas[key] = schema
            self._fingerprints[key] = fingerprint or fingerprint_schemas(schema)
        return key

    def get(self, source: SourceArgType,
            schema_class: Optional[type[SchemaType]] = None,
            **kwargs: Any) -> SchemaType:
        """
        Returns the registered schema for *source*, building and registering one
        on first use. Extra keyword arguments are passed to the schema constructor
        only when the schema has to be created.
        """
        if isinstance(source, XMLResource):
            key = source.url or f'id:{id(source)}'
        else:
            url = get_url(source)
            if url is None:
                # In-memory sources can't be keyed stably across calls: register
                # them explicitly instead of silently creating duplicate instances.
                msg = "only URL or XMLResource sources can be fetched, " \
                      "use register() for in-memory schemas"
                raise XMLSchemaReloadError(msg)
            # Normalize file paths to file:// URLs so that the same resource
            # passed as path or as URI maps to the same registry entry.
            key = normalize_url(url)

        with self._lock:
            if key in self._schemas:
                return self._schemas[key]

            cls = schema_class or self.schema_class
            if cls is None:
                from xmlschema.validators import XMLSchema10
                cls = XMLSchema10

            if self.settings is not None:
                schema = cls.from_settings(self.settings, source, **kwargs)
            else:
                schema = cls(source, **kwargs)

            if key in self._schemas:
                # Another thread may have won the race while the schema was building
                return self._schemas[key]

            self._schemas[key] = schema
            self._fingerprints[key] = fingerprint_schemas(schema)
            return schema

    def _resolve_key(self, location: str) -> str:
        key = normalize_url(location)
        if key not in self._schemas:
            raise KeyError(location)
        return key

    def fingerprint(self, location: str) -> SchemaFingerprint:
        """Returns the fingerprint recorded for the schema at *location*."""
        with self._lock:
            return self._fingerprints[self._resolve_key(location)]

    def versions(self) -> dict[str, str]:
        """
        Returns a mapping of each registered location to its short schema
        fingerprint, useful for logging or exposing which rules are in use.
        """
        with self._lock:
            return {key: fp.short for key, fp in self._fingerprints.items()}

    def is_stale(self, location: str, check_content: bool = False) -> bool:
        """
        Returns ``True`` if any XSD resource of the schema registered at
        *location* changed since its fingerprint was recorded.

        :raises KeyError: if *location* is not registered.
        :raises XMLResourceOSError: if one of the resource URLs is not accessible.
        """
        with self._lock:
            location = self._resolve_key(location)
            schema = self._schemas[location]
            current = {f.url: f for f in self._fingerprints[location].sources}
            for other in schema.maps.iter_schemas():
                if _is_library_schema(other):
                    continue
                recorded = current.get(other.source.url)
                if recorded is None:
                    return True
                if resource_is_stale(other.source, recorded, check_content):
                    return True
            return False

    def reload(self, location: Optional[str] = None,
               check_content: bool = False,
               force: bool = False) -> list[str]:
        """
        Reloads schemas whose XSD sources changed, swapping in the new rules
        atomically. Registered in-memory schemas are skipped (they cannot be
        re-read from a location).

        :param location: limits the reload to a single registered location. When
            omitted every registered location is checked.
        :param check_content: re-read and hash every resource instead of relying
            on local file modification times.
        :param force: reload even if no change is detected.
        :returns: the list of locations that were actually reloaded.
        :raises XMLSchemaReloadError: if building the new schema fails; the
            previously active schema keeps serving in that case.
        """
        with self._lock:
            if location is None:
                targets = list(self._schemas)
            else:
                targets = [self._resolve_key(location)]

            reloaded: list[str] = []
            for key in targets:
                old = self._schemas[key]
                if old.source.url is None:
                    continue
                if not force and not self.is_stale(key, check_content):
                    continue

                cls = type(old)
                try:
                    new_schema = cls(old.source.url)
                    new_schema.build()
                except Exception as err:
                    if isinstance(err, XMLSchemaReloadError):
                        raise
                    raise XMLSchemaReloadError(
                        f"reload of schema {key!r} failed, "
                        f"keeping the previous version: {err}"
                    ) from err

                # The old object is updated in place, so every reference cached
                # by application code starts using the new rules.
                old.reload_from(new_schema)
                self._fingerprints[key] = fingerprint_schemas(old)
                reloaded.append(key)

            return reloaded

    def clear(self) -> None:
        """Removes every registered schema."""
        with self._lock:
            self._schemas.clear()
            self._fingerprints.clear()


#: Process-wide default registry, for applications that want a single shared
#: point of schema management.
registry = SchemaRegistry()
