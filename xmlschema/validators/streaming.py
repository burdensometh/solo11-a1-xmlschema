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
Streaming XML validation based on ElementTree's iterparse.

Large XML documents can't be fully loaded into an in-memory Element tree without
exhausting the available memory, and pre-splitting the document into chunks
breaks its structure. This module validates a document while it's being parsed:

* each element is validated as soon as its closing tag is reached, then the
  element is cleared so that its memory can be reclaimed;
* cross-node constraints that can't be decided on a single element (xs:ID
  uniqueness and dangling xs:IDREF references) are tracked with lightweight
  sets of scalar values instead of retaining elements;
* the process memory footprint is checked periodically while parsing and, if a
  configurable threshold is exceeded, validation aborts immediately with an
  :class:`~xmlschema.exceptions.XMLResourceExceeded` instead of growing until
  the operating system kills the process.

Only checks that can be decided locally (element declaration lookup, attribute
and simple-content typing, required attributes, xs:ID/xs:IDREF constraints) are
performed: full XSD model/identity-constraint validation needs the complete tree
and is still provided by :meth:`XMLSchema.validate`. Think of this module as a
memory-bounded pre-check to reject invalid or unmanageably large documents
before running a full validation.
"""
import gc
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Optional, cast
from xml.etree import ElementTree

import xmlschema.names as nm
from xmlschema import _limits
from xmlschema.aliases import ElementType, IOType, NsmapType, SchemaElementType
from xmlschema.exceptions import XMLResourceExceeded, \
    XMLResourceParseError, XMLSchemaTypeError
from xmlschema.namespaces import NamespaceMapper
from xmlschema.resources import XMLResource
from xmlschema.resources.xml_resource import XMLResourceManager
from xmlschema.translation import gettext as _

from .attributes import XsdAttribute
from .elements import XsdElement
from .exceptions import XMLSchemaValidationError
from .simple_types import XsdSimpleType
from .validation import ValidationContext
from .wildcards import XsdAnyElement
from .xsdbase import XsdValidator

if TYPE_CHECKING:
    from .schemas import XMLSchemaBase

__all__ = ['StreamValidationError', 'MemoryLimitError', 'MemoryMonitor',
           'StreamingValidator', 'iter_stream_errors', 'stream_validate',
           'is_stream_valid']


# Memory check period: an RSS probe on each parsed element would dominate the
# runtime, so the footprint is sampled every N elements.
_MEMORY_CHECK_INTERVAL = 1024


def _is_seekable(fp: object) -> bool:
    """Returns `True` if *fp* can be rewound and re-read."""
    seekable = getattr(fp, 'seekable', None)
    if seekable is None:
        return False
    try:
        return bool(seekable())
    except (OSError, ValueError):
        return False


def _spool_stream(fp: IOType) -> IOType:
    """
    Copies a non-seekable file-like object into a spooled temporary file.

    The returned binary stream is seekable and owns its file: it is closed and
    released by the caller. Spooling keeps the original stream's memory
    footprint bounded (the data lands on disk once it exceeds the spool size).
    """
    tmp: IOType = tempfile.SpooledTemporaryFile(
        max_size=1024 * 1024, mode='w+b'
    )
    try:
        shutil.copyfileobj(cast(Any, fp), cast(Any, tmp))
    except BaseException:
        tmp.close()
        raise
    tmp.seek(0)
    return tmp


class StreamValidationError(XMLSchemaValidationError):
    """A validation error found during streaming validation."""


class MemoryLimitError(XMLResourceExceeded):
    """
    Raised when the process memory footprint exceeds the threshold configured
    on a :class:`StreamingValidator` while parsing the XML data.
    """


class MemoryMonitor:
    """
    Tracks the process resident memory while parsing and enforces a limit.

    :param max_bytes: maximum resident memory allowed, in bytes. The check is
        disabled if *max_bytes* is `None` or not positive.
    :param check_interval: number of elements parsed between two memory checks.
    """
    __slots__ = ('max_bytes', 'check_interval', 'elements_since_check',
                 '_get_rss', '_fallback_rss')

    def __init__(self, max_bytes: Optional[int] = None,
                 check_interval: int = _MEMORY_CHECK_INTERVAL) -> None:
        if max_bytes is not None and (
                not isinstance(max_bytes, int) or max_bytes < 0):
            raise XMLSchemaTypeError(
                _("max_bytes must be a non-negative int or None, got {!r}").format(max_bytes)
            )
        if not isinstance(check_interval, int) or check_interval < 1:
            raise XMLSchemaTypeError(
                _("check_interval must be an int >= 1, got {!r}").format(check_interval)
            )

        self.max_bytes = max_bytes
        self.check_interval = check_interval
        self.elements_since_check = 0
        self._get_rss: Optional[Callable[[], int]] = None
        self._fallback_rss = 0
        self._setup_probe()

    def _setup_probe(self) -> None:
        """Selects the cheapest available RSS probe for this platform."""
        try:
            import resource
        except ImportError:
            resource = None  # type: ignore[assignment]

        # /proc/self/statm reports the *current* RSS (not the peak), which is
        # the meaningful measure for a process that continuously frees memory.
        if sys.platform.startswith('linux'):
            statm_path = '/proc/self/statm'
            page_size = self._get_page_size()

            def get_rss() -> int:
                with open(statm_path, 'rb') as fp:
                    rss_pages = int(fp.read().split()[1])
                return rss_pages * page_size

            self._get_rss = get_rss
            return

        if resource is not None:
            # On Linux ru_maxrss is kilobytes, on macOS bytes. It is a peak
            # measure, used only where /proc is not available.
            scale = 1 if sys.platform == 'darwin' else 1024

            def get_rss() -> int:
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale

            self._get_rss = get_rss

    @staticmethod
    def _get_page_size() -> int:
        try:
            import os
            return os.sysconf('SC_PAGE_SIZE')
        except (AttributeError, ValueError, OSError):
            return 4096

    @property
    def enabled(self) -> bool:
        """Returns `True` if a memory limit has to be enforced."""
        return bool(self.max_bytes)

    def current_rss(self) -> int:
        """Returns the current resident set size of the process, in bytes."""
        if self._get_rss is not None:
            return self._get_rss()
        return self._fallback_rss

    def check(self, allocated_bytes: int = 0) -> None:
        """
        Verifies the memory limit after another parsed element.

        :param allocated_bytes: coarse estimate of the memory owned by the
            streaming validator, used when no OS-level probe is available.
        :raises MemoryLimitError: if the resident memory exceeds *max_bytes*.
        """
        if not self.enabled:
            return

        self.elements_since_check += 1
        if self.elements_since_check < self.check_interval:
            return
        self.elements_since_check = 0

        if self._get_rss is not None:
            rss = self._get_rss()
        else:
            gc.collect()
            self._fallback_rss = max(self._fallback_rss, allocated_bytes)
            rss = self._fallback_rss

        limit = self.max_bytes
        if limit is not None and rss > limit:
            msg = _("memory limit of {0} bytes exceeded while parsing "
                    "(current RSS {1} bytes)")
            raise MemoryLimitError(msg.format(limit, rss))


class StreamingValidator:
    """
    A memory-bounded, iterparse-based XML validator.

    The validator drives an ``ElementTree.iterparse`` iterator itself: on each
    ``end`` event the element is checked against the schema and immediately
    cleared, keeping the retained tree at roughly the depth of the document
    instead of its size. Cross-node constraints (xs:ID uniqueness, dangling
    xs:IDREF) are maintained with lightweight sets of scalar values in place of
    retained elements.

    :param schema: a built XSD schema used to resolve global elements and types.
    :param max_memory_bytes: process RSS threshold in bytes. When exceeded a
        :class:`MemoryLimitError` is raised. `None` (default) disables the check.
    :param namespaces: optional prefix/namespace map augmenting the declarations
        found in the XML data.
    :param validation: validation mode, ``'lax'`` (default) collects errors,
        ``'strict'`` raises at the first error. ``'skip'`` only parses.
    :param memory_check_interval: number of elements between two RSS checks.
    """

    def __init__(self,
                 schema: 'XMLSchemaBase',
                 max_memory_bytes: Optional[int] = None,
                 namespaces: Optional[NsmapType] = None,
                 validation: str = 'lax',
                 memory_check_interval: int = _MEMORY_CHECK_INTERVAL) -> None:

        if validation not in ('lax', 'strict', 'skip'):
            raise XMLSchemaTypeError(
                _("validation must be 'lax', 'strict' or 'skip'")
            )

        self.schema: 'XMLSchemaBase' = schema
        self.validation = validation
        self.memory_monitor = MemoryMonitor(max_memory_bytes, memory_check_interval)

        # Lightweight cross-node state: only scalar values are retained.
        self.ids: set[str] = set()
        self.idrefs: set[str] = set()
        self.duplicate_ids: set[str] = set()

        self._namespaces = dict(namespaces) if namespaces else {}
        self._element_count = 0
        self._max_depth = 0

        # Ancestor tags (braced QNames) and the matching XSD elements drive
        # local element resolution. The child cache maps a parent component
        # plus a tag to the resolved component, `False` if unknown. Both are
        # bounded by the tree depth and the schema model size, never by the
        # number of document elements.
        self._ancestors: list[str] = []
        self._xsd_stack: list[Optional[SchemaElementType]] = []
        self._child_cache: dict[tuple[SchemaElementType, str], object] = {}

        # Set when a non-seekable input stream is spooled to a temp buffer;
        # closed at the end of iter_errors().
        self._spooled_fp: Optional[IOType] = None

    @property
    def element_count(self) -> int:
        """Number of elements processed so far."""
        return self._element_count

    @property
    def max_depth(self) -> int:
        """Maximum element depth reached while parsing."""
        return self._max_depth

    #
    # Public API
    #

    def _get_resource(self, source: object):
        """
        Builds a *lazy* :class:`XMLResource` for *source*.

        A lazy resource only materializes the root element at construction, so
        the whole document is never held in memory; this validator then drives
        its own ``iterparse`` over the (reopened) stream. A resource created
        from an already parsed Element/ElementTree can't be turned lazy: in
        that case the tree is already in memory and the resource is reused.

        A non-seekable file-like object is spooled to a temporary seekable
        buffer first, otherwise the bytes consumed by the root probe could not
        be re-read by the main parser.
        """
        settings = self.schema.maps.settings
        spooled: Optional[IOType] = None

        if isinstance(source, XMLResource):
            if source.is_lazy():
                return source
            # Re-openable representations can be streamed again; otherwise the
            # document only survives as the already materialized Element tree.
            reopenable = source.fp is not None or source.url is not None \
                or source.text is not None
            if not reopenable:
                return source
            # Re-open from the most concrete representation available.
            source = source.fp or source.url or source.text

        if hasattr(source, 'read') and not _is_seekable(source):
            spooled = _spool_stream(cast(IOType, source))
            source = spooled
        self._spooled_fp = spooled

        return settings.get_resource(XMLResource, cast(Any, source), lazy=True)

    def iter_errors(self, source: object) -> Iterator[XMLSchemaValidationError]:
        """
        Lazily parses and validates *source*, yielding validation errors.

        :param source: a path/URL string, a file-like object or an
            :class:`~xmlschema.XMLResource` providing the XML data. Element
            trees and plain XML strings are accepted but, being already fully
            materialized, defeat the purpose of streaming validation.
        """
        resource = self._get_resource(source)

        context = self._make_context(resource)
        depth = 0
        remaining_levels = _limits.MAX_XML_DEPTH
        self._ancestors.clear()
        self._xsd_stack.clear()
        self._child_cache.clear()
        # Reset per-document state so one validator can be reused safely.
        self.ids.clear()
        self.idrefs.clear()
        self.duplicate_ids.clear()
        self._element_count = 0
        self._max_depth = 0

        try:
            if not resource.is_lazy():
                # No stream available (Element/ElementTree or an opaque, fully
                # consumed source): walk the materialized tree with the same
                # lightweight local checks.
                yield from self._iter_tree_errors(resource.root, context)
                return

            with XMLResourceManager(resource) as cm:
                try:
                    iterator = ElementTree.iterparse(
                        cm.fp, events=('start', 'end', 'start-ns', 'end-ns')
                    )
                    for event, node in iterator:
                        if event == 'start-ns':
                            prefix, uri = node
                            self._namespaces.setdefault(prefix or '', uri)
                            continue
                        elif event == 'end-ns':
                            continue
                        elif event == 'start':
                            remaining_levels -= 1
                            if not remaining_levels:
                                msg = ("maximum XML depth reached "
                                       "(MAX_XML_DEPTH={}) while streaming {}")
                                raise XMLResourceExceeded(
                                    msg.format(_limits.MAX_XML_DEPTH, resource)
                                )
                            depth += 1
                            if depth > self._max_depth:
                                self._max_depth = depth
                            self._ancestors.append(node.tag)
                            # Resolve the component once, while the parent is
                            # still on the stack; reuse it at the end event.
                            self._xsd_stack.append(
                                self._resolve_element(node, depth == 1)
                            )
                            continue

                        # event == 'end': the subtree of node is complete.
                        xsd_element = self._xsd_stack.pop()
                        remaining_levels += 1
                        depth -= 1
                        self._element_count += 1
                        self.memory_monitor.check(self._retained_estimate())

                        self._validate_element(node, xsd_element, context)
                        yield from context.errors
                        context.errors.clear()

                        # Release the validated subtree straight away. Children
                        # are removed (tag/attributes survive until the element's
                        # own parent is processed) and bookkeeping is pruned.
                        self._release_element(resource, node)
                        self._ancestors.pop()

                except ElementTree.ParseError as err:
                    raise XMLResourceParseError(
                        _("invalid XML syntax: {}").format(err)
                    ) from err

            # The whole document has been seen: dangling IDREFs can now be
            # decided. Kept outside any finally block (a yield there is illegal
            # when the generator is closed early, e.g. by is_valid()).
            yield from self._check_dangling_idrefs(context)
        finally:
            if self._spooled_fp is not None:
                self._spooled_fp.close()
                self._spooled_fp = None

    def _iter_tree_errors(self, root: ElementType,
                          context: ValidationContext) -> Iterator[XMLSchemaValidationError]:
        """Validation path for an already materialized Element tree."""
        def walk(elem: ElementType, is_root: bool) -> None:
            self._ancestors.append(elem.tag)
            self._max_depth = max(self._max_depth, len(self._ancestors))
            self._element_count += 1
            xsd_element = self._resolve_element(elem, is_root)
            self._xsd_stack.append(xsd_element)
            self._validate_element(elem, xsd_element, context)
            for child in list(elem):
                walk(child, False)
            self._xsd_stack.pop()
            self._ancestors.pop()

        walk(root, True)
        yield from context.errors
        context.errors.clear()
        yield from self._check_dangling_idrefs(context)

    def validate(self, source: object) -> None:
        """
        Streams *source* and raises the first found validation error.

        :raises XMLSchemaValidationError: if the document is invalid.
        :raises MemoryLimitError: if the memory threshold is exceeded.
        """
        for error in self.iter_errors(source):
            raise error

    def is_valid(self, source: object) -> bool:
        """Returns `True` if streaming validation finds no error."""
        return next(self.iter_errors(source), None) is None

    #
    # Internal helpers
    #

    def _make_context(self, resource: object) -> ValidationContext:
        context = ValidationContext(
            source=resource,
            converter=NamespaceMapper(self._namespaces, source=resource),
            # ID/IDREF uniqueness and resolution are tracked by this validator
            # with lightweight sets; the schema's own id_map bookkeeping would
            # retain the same scalar values and raise before the sets are seen.
            check_identities=False,
        )
        # Keep a non-zero level so component decoding behaves as in a document.
        context.level = 1
        return context

    def _retained_estimate(self) -> int:
        # Rough estimate of cross-node state, used only when no OS-level RSS
        # probe is available (each set entry is a short string object).
        return (len(self.ids) + len(self.idrefs) + len(self.duplicate_ids)) * 64

    def _add_error(self, validator: XsdValidator, obj: object, reason: str,
                   context: ValidationContext) -> None:
        """Records an error, raising immediately in strict mode."""
        error = StreamValidationError(
            validator, obj, reason, context.source, context.namespaces
        )
        if self.validation == 'strict':
            raise error
        if self.validation != 'skip':
            context.errors.append(error)

    def _resolve_element(self, elem: ElementType,
                         is_root: bool) -> Optional[SchemaElementType]:
        """
        Resolves the XSD element matching *elem* incrementally.

        The root is taken from the schema's global element map; every other
        element is looked up inside the *content model* of the XSD element on
        top of the ancestor stack. Results are cached per (parent component,
        tag) pair, so the cost is paid once per distinct schema location rather
        than on every document node, and no full-path XPath has to be parsed.
        """
        if is_root:
            return self.schema.maps.elements.get(elem.tag)

        parent = self._xsd_stack[-1] if self._xsd_stack else None
        if parent is None:
            return self.schema.maps.elements.get(elem.tag)

        if isinstance(parent, XsdAnyElement):
            # A wildcard does not define a content model: its descendants are
            # accepted without local typing (full checking of processContents
            # belongs to the complete-tree validator).
            return parent

        cache_key = (parent, elem.tag)
        try:
            cached = self._child_cache[cache_key]
        except KeyError:
            pass
        else:
            return None if cached is False else cast(SchemaElementType, cached)

        xsd_element = self._find_child_element(parent, elem.tag)
        if xsd_element is None:
            # Fall back to the global declaration so a globally defined
            # element used in a local context is still checked.
            xsd_element = self.schema.maps.elements.get(elem.tag)

        self._child_cache[cache_key] = xsd_element if xsd_element is not None else False
        return xsd_element

    @staticmethod
    def _find_child_element(parent: XsdElement,
                            tag: str) -> Optional[SchemaElementType]:
        """
        Searches a child element declaration by tag inside a parent model.

        Returns the matching :class:`XsdElement`, an :class:`XsdAnyElement`
        wildcard that accepts the tag, or `None` if the model declares neither.
        """
        xsd_type = parent.type
        content = getattr(xsd_type, 'content', None)
        # A complex type exposes its particles through the content group; a
        # simple type has no element children.
        iter_elements = getattr(content, 'iter_elements', None)
        if iter_elements is None:
            return None

        default_ns = parent.target_namespace if '{' not in tag else None
        wildcard: Optional[XsdAnyElement] = None
        for child in iter_elements():
            if isinstance(child, XsdAnyElement):
                if wildcard is None and child.is_matching(tag, default_ns):
                    wildcard = child
                continue
            if child.name == tag:
                ref = getattr(child, 'ref', None)
                return ref if ref is not None else child
        return wildcard

    def _validate_element(self, elem: ElementType,
                          xsd_element: Optional[SchemaElementType],
                          context: ValidationContext) -> None:
        if xsd_element is None:
            if self.validation != 'skip':
                reason = _("{!r} is not an element of the schema").format(elem.tag)
                self._add_error(self.schema, elem, reason, context)
            return

        if isinstance(xsd_element, XsdAnyElement):
            # An xs:any wildcard accepts the element; full content typing of
            # wildcard data is left to the complete-tree validator.
            return

        xsd_type = xsd_element.type
        attribute_group = xsd_element.get_attributes(xsd_type)

        # Required attributes (local check, no tree state needed).
        if self.validation != 'skip':
            for name in attribute_group.iter_required():
                if name not in elem.attrib:
                    reason = _("missing required attribute {!r}").format(name)
                    self._add_error(xsd_element, elem, reason, context)

        # Attribute typing plus ID/IDREF tracking.
        for attr_name, attr_value in elem.attrib.items():
            xsd_attribute = attribute_group.get(attr_name)
            if not isinstance(xsd_attribute, XsdAttribute):
                continue  # anyAttribute or XSI attributes: handled by the schema
            attr_type = xsd_attribute.type
            if not isinstance(attr_type, XsdSimpleType):
                continue

            decoded = self._check_simple_value(
                attr_type, attr_value, xsd_attribute, context
            )
            if decoded is None:
                continue

            # Check the attribute's own type, not its primitive root type:
            # xs:ID/xs:IDREF root type is xs:string.
            if attr_type.is_key():
                self._register_id(decoded, xsd_attribute, attr_value, context)
            elif self._is_idref(attr_type):
                # References are kept even if the target ID was already seen:
                # dangling references can only be decided at document end.
                self.idrefs.add(str(decoded))

        # Simple-content element typing. Element-only content is not checked
        # here: full model validation requires retaining the children.
        if elem.text is not None and isinstance(xsd_type, XsdSimpleType):
            text = elem.text
            decoded = self._check_simple_value(xsd_type, text, xsd_element, context)
            if decoded is not None:
                if xsd_type.is_key():
                    self._register_id(decoded, xsd_element, text, context)
                elif self._is_idref(xsd_type):
                    self.idrefs.add(str(decoded))

    @staticmethod
    def _is_idref(xsd_type: XsdSimpleType) -> bool:
        """Returns `True` if *xsd_type* is xs:IDREF or a type derived from it."""
        idref_type = xsd_type.maps.types.get(nm.XSD_IDREF)
        return idref_type is not None and xsd_type.is_derived(idref_type)

    def _check_simple_value(self, xsd_type: XsdSimpleType, text: str,
                            validator: XsdValidator,
                            context: ValidationContext) -> Optional[object]:
        """
        Validates a simple value in lax mode against *xsd_type*.

        Returns the decoded value, or `None` when the value is invalid (the
        validation error is appended to the context error list). In strict
        mode the first invalid value raises.
        """
        if self.validation == 'skip':
            try:
                return xsd_type.text_decode(text, 'skip', context)
            except (ValueError, TypeError):
                return None

        errors_before = len(context.errors)
        try:
            decoded = xsd_type.text_decode(text, 'lax', context)
        except XMLSchemaValidationError as err:
            reason = err.reason
        else:
            new_errors = context.errors[errors_before:]
            del context.errors[errors_before:]
            reason = new_errors[0].reason if new_errors else None

        if reason is not None:
            error = StreamValidationError(
                validator, text, reason, context.source, context.namespaces
            )
            if self.validation == 'strict':
                raise error
            context.errors.append(error)
            return None
        return decoded

    def _register_id(self, value: object, validator: XsdValidator, obj: object,
                     context: ValidationContext) -> None:
        """Registers an xs:ID value, flagging duplicates in lightweight sets."""
        key = str(value)
        if key in self.ids:
            self.duplicate_ids.add(key)
            if self.validation != 'skip':
                reason = _("duplicated xs:ID value {!r}").format(key)
                self._add_error(validator, obj, reason, context)
        else:
            self.ids.add(key)

    def _check_dangling_idrefs(self,
                               context: ValidationContext) \
            -> Iterator[XMLSchemaValidationError]:
        if self.validation != 'skip':
            for value in sorted(self.idrefs):
                if value not in self.ids:
                    reason = _("IDREF %r not found in XML document") % value
                    self._add_error(self.schema, value, reason, context)
        yield from context.errors
        context.errors.clear()

    @staticmethod
    def _release_element(resource: object, elem: ElementType) -> None:
        """
        Drops a validated element subtree and its parser bookkeeping.

        Uses ``del elem[:]`` (linear) rather than repeated ``remove`` calls
        (quadratic for an element with many siblings), so a container with
        hundreds of thousands of children is released in linear time. The
        element keeps its own tag/attributes until its parent end event prunes
        it; retained state therefore stays proportional to the tree depth.
        """
        nsmaps = getattr(resource, '_nsmaps', None)
        xmlns = getattr(resource, '_xmlns', None)
        if nsmaps is not None:
            for node in elem.iter():
                nsmaps.pop(node, None)
                if xmlns is not None:
                    xmlns.pop(node, None)
        del elem[:]


def iter_stream_errors(schema: 'XMLSchemaBase', source: object,
                       max_memory_bytes: Optional[int] = None,
                       namespaces: Optional[NsmapType] = None,
                       memory_check_interval: int = _MEMORY_CHECK_INTERVAL
                       ) -> Iterator[XMLSchemaValidationError]:
    """Convenience wrapper around :class:`StreamingValidator` yielding errors."""
    validator = StreamingValidator(
        schema,
        max_memory_bytes=max_memory_bytes,
        namespaces=namespaces,
        validation='lax',
        memory_check_interval=memory_check_interval,
    )
    yield from validator.iter_errors(source)


def stream_validate(schema: 'XMLSchemaBase', source: object,
                    max_memory_bytes: Optional[int] = None,
                    namespaces: Optional[NsmapType] = None) -> None:
    """Convenience wrapper around :class:`StreamingValidator` raising on error."""
    validator = StreamingValidator(
        schema,
        max_memory_bytes=max_memory_bytes,
        namespaces=namespaces,
        validation='strict',
    )
    validator.validate(source)


def is_stream_valid(schema: 'XMLSchemaBase', source: object,
                    max_memory_bytes: Optional[int] = None,
                    namespaces: Optional[NsmapType] = None) -> bool:
    """Convenience wrapper around :class:`StreamingValidator` returning a bool."""
    validator = StreamingValidator(
        schema,
        max_memory_bytes=max_memory_bytes,
        namespaces=namespaces,
        validation='lax',
    )
    return validator.is_valid(source)
