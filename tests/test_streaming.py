#!/usr/bin/env python
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
Tests for the iterparse-based streaming validator
(:mod:`xmlschema.validators.streaming`).
"""
import io
import pathlib
import tempfile
import unittest
from textwrap import dedent

import xmlschema
from xmlschema import XMLResource
from xmlschema.exceptions import XMLResourceExceeded, XMLResourceParseError, \
    XMLSchemaTypeError
from xmlschema.validators import StreamingValidator, MemoryMonitor, \
    MemoryLimitError, StreamValidationError, iter_stream_errors, \
    stream_validate, is_stream_valid


ID_SCHEMA = dedent("""\
    <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
      <xs:element name="root">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="item" maxOccurs="unbounded">
              <xs:complexType>
                <xs:simpleContent>
                  <xs:extension base="xs:string">
                    <xs:attribute name="id" type="xs:ID" use="required"/>
                    <xs:attribute name="ref" type="xs:IDREF"/>
                    <xs:attribute name="n" type="xs:int"/>
                  </xs:extension>
                </xs:simpleContent>
              </xs:complexType>
            </xs:element>
          </xs:sequence>
        </xs:complexType>
      </xs:element>
    </xs:schema>""")

SIMPLE_CONTENT_SCHEMA = dedent("""\
    <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
      <xs:element name="root">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="k" type="xs:ID" maxOccurs="unbounded"/>
            <xs:element name="r" type="xs:IDREF" maxOccurs="unbounded"/>
          </xs:sequence>
        </xs:complexType>
      </xs:element>
    </xs:schema>""")


class NonSeekableStream(io.RawIOBase):
    """A byte stream that reports itself as non-seekable."""

    def __init__(self, data: bytes, chunk_size: int = 64):
        self._data = data
        self._pos = 0
        self._chunk_size = chunk_size
        self.closed_by_validator = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b):
        if self._pos >= len(self._data):
            return 0
        n = min(self._chunk_size, len(b), len(self._data) - self._pos)
        b[:n] = self._data[self._pos:self._pos + n]
        self._pos += n
        return n

    def close(self) -> None:
        self.closed_by_validator = True
        super().close()


class TestMemoryMonitor(unittest.TestCase):

    def test_disabled_by_default(self):
        monitor = MemoryMonitor()
        self.assertIsNone(monitor.max_bytes)
        self.assertFalse(monitor.enabled)
        # Must never raise when disabled, regardless of the counter.
        for _ in range(5000):
            monitor.check()

    def test_invalid_arguments(self):
        with self.assertRaises(XMLSchemaTypeError):
            MemoryMonitor(max_bytes=-1)
        with self.assertRaises(XMLSchemaTypeError):
            MemoryMonitor(max_bytes='100')
        with self.assertRaises(XMLSchemaTypeError):
            MemoryMonitor(check_interval=0)

    def test_current_rss_is_positive(self):
        monitor = MemoryMonitor()
        self.assertGreater(monitor.current_rss(), 0)

    def test_limit_exceeded(self):
        # One megabyte is below the footprint of an interpreter that has
        # imported the xmlschema package, so the check must trip promptly.
        monitor = MemoryMonitor(max_bytes=1024 * 1024, check_interval=1)
        with self.assertRaises(MemoryLimitError):
            for _ in range(100):
                monitor.check()
        self.assertIsInstance(MemoryLimitError(), XMLResourceExceeded)

    def test_limit_not_exceeded_when_high(self):
        monitor = MemoryMonitor(max_bytes=10 ** 12, check_interval=1)
        for _ in range(10):
            monitor.check()  # no exception


class TestStreamingValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.schema = xmlschema.XMLSchema(ID_SCHEMA)
        cls.simple_schema = xmlschema.XMLSchema(SIMPLE_CONTENT_SCHEMA)

    def test_invalid_validation_mode(self):
        with self.assertRaises(XMLSchemaTypeError):
            StreamingValidator(self.schema, validation='bogus')

    # -- element / attribute local checks ---------------------------------

    def test_valid_document(self):
        xml = '<root><item id="a" n="1">x</item>' \
              '<item id="b" ref="a" n="2">y</item></root>'
        validator = StreamingValidator(self.schema)
        self.assertEqual([], list(validator.iter_errors(io.StringIO(xml))))
        self.assertEqual(validator.element_count, 3)
        self.assertEqual(validator.max_depth, 2)
        self.assertEqual(validator.ids, {'a', 'b'})
        self.assertEqual(validator.idrefs, {'a'})

    def test_unknown_element(self):
        validator = StreamingValidator(self.schema)
        errors = list(validator.iter_errors(io.StringIO('<root><other/></root>')))
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], StreamValidationError)
        self.assertIn('not an element of the schema', errors[0].reason)

    def test_invalid_attribute_type(self):
        xml = '<root><item id="a" n="not-an-int">x</item></root>'
        validator = StreamingValidator(self.schema)
        errors = list(validator.iter_errors(io.StringIO(xml)))
        self.assertEqual(1, len(errors))
        self.assertIn('int', errors[0].reason)

    def test_missing_required_attribute(self):
        validator = StreamingValidator(self.schema)
        errors = list(validator.iter_errors(io.StringIO('<root><item>x</item></root>')))
        self.assertEqual(1, len(errors))
        self.assertIn("missing required attribute 'id'", errors[0].reason)

    def test_strict_mode_raises(self):
        validator = StreamingValidator(self.schema, validation='strict')
        with self.assertRaises(StreamValidationError):
            validator.validate(io.StringIO('<root><item>x</item></root>'))

    def test_skip_mode_collects_nothing(self):
        xml = '<root><item id="a" n="bad">x</item><other/></root>'
        validator = StreamingValidator(self.schema, validation='skip')
        self.assertEqual([], list(validator.iter_errors(io.StringIO(xml))))

    def test_is_valid_helper(self):
        good = '<root><item id="a" n="1">x</item></root>'
        bad = '<root><item n="1">x</item></root>'
        self.assertTrue(is_stream_valid(self.schema, io.StringIO(good)))
        self.assertFalse(is_stream_valid(self.schema, io.StringIO(bad)))

    def test_module_level_functions(self):
        good = '<root><item id="a" n="1">x</item></root>'
        bad = '<root><item id="a" n="bad">x</item></root>'
        stream_validate(self.schema, io.StringIO(good))  # no exception
        self.assertEqual(
            [], list(iter_stream_errors(self.schema, io.StringIO(good)))
        )
        self.assertEqual(
            1, len(list(iter_stream_errors(self.schema, io.StringIO(bad))))
        )

    # -- ID / IDREF cross-node constraints --------------------------------

    def test_duplicate_id_attribute(self):
        xml = '<root><item id="a"/><item id="a"/></root>'
        validator = StreamingValidator(self.schema)
        errors = list(validator.iter_errors(io.StringIO(xml)))
        self.assertEqual(1, len(errors))
        self.assertIn("duplicated xs:ID value 'a'", errors[0].reason)
        self.assertEqual(validator.duplicate_ids, {'a'})

    def test_dangling_idref(self):
        xml = '<root><item id="a" ref="missing">x</item></root>'
        validator = StreamingValidator(self.schema)
        errors = list(validator.iter_errors(io.StringIO(xml)))
        self.assertEqual(1, len(errors))
        self.assertIn("IDREF 'missing' not found", errors[0].reason)

    def test_forward_and_backward_idref(self):
        forward = '<root><item id="a" ref="b">x</item><item id="b">y</item></root>'
        backward = '<root><item id="b">y</item><item id="a" ref="b">x</item></root>'
        for xml in (forward, backward):
            validator = StreamingValidator(self.schema)
            self.assertEqual([], list(validator.iter_errors(io.StringIO(xml))))

    def test_simple_content_id_and_idref(self):
        ok = '<root><k>x</k><k>y</k><r>x</r></root>'
        validator = StreamingValidator(self.simple_schema)
        self.assertTrue(validator.is_valid(io.StringIO(ok)))

        dup = '<root><k>x</k><k>x</k></root>'
        validator = StreamingValidator(self.simple_schema)
        errors = list(validator.iter_errors(io.StringIO(dup)))
        self.assertEqual(1, len(errors))
        self.assertIn("duplicated xs:ID value 'x'", errors[0].reason)

        dangling = '<root><k>x</k><r>q</r></root>'
        validator = StreamingValidator(self.simple_schema)
        errors = list(validator.iter_errors(io.StringIO(dangling)))
        self.assertEqual(1, len(errors))
        self.assertIn("IDREF 'q' not found", errors[0].reason)

    def test_strict_dangling_idref_raises_at_end(self):
        validator = StreamingValidator(self.schema, validation='strict')
        with self.assertRaises(StreamValidationError):
            validator.validate(
                io.StringIO('<root><item id="a" ref="missing">x</item></root>')
            )

    # -- source handling ---------------------------------------------------

    def test_bytes_io_source(self):
        xml = b'<root><item id="a" n="1">x</item></root>'
        validator = StreamingValidator(self.schema)
        self.assertTrue(validator.is_valid(io.BytesIO(xml)))

    def test_string_source(self):
        xml = '<root><item id="a" n="1">x</item></root>'
        validator = StreamingValidator(self.schema)
        self.assertTrue(validator.is_valid(xml))

    def test_file_path_source(self):
        xml = '<root><item id="a" n="1">x</item></root>'
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'data.xml'
            path.write_text(xml)
            validator = StreamingValidator(self.schema)
            self.assertTrue(validator.is_valid(str(path)))

    def test_non_seekable_stream_is_spooled(self):
        xml = b'<root><item id="a" ref="b">x</item><item id="b">y</item></root>'
        stream = NonSeekableStream(xml)
        validator = StreamingValidator(self.schema)
        errors = list(validator.iter_errors(stream))
        self.assertEqual([], errors)
        # The spooled temporary buffer is released at the end.
        self.assertIsNone(validator._spooled_fp)

    def test_non_seekable_stream_malformed(self):
        stream = NonSeekableStream(b'<root><item id="a"></root>')
        validator = StreamingValidator(self.schema)
        with self.assertRaises(XMLResourceParseError):
            list(validator.iter_errors(stream))

    def test_lazy_xml_resource_source(self):
        xml = '<root><item id="a" n="1">x</item></root>'
        resource = XMLResource(io.StringIO(xml), lazy=True)
        validator = StreamingValidator(self.schema)
        self.assertTrue(validator.is_valid(resource))

    def test_materialized_element_source(self):
        import xml.etree.ElementTree as ET
        tree = ET.fromstring('<root><item id="a" n="1">x</item></root>')
        resource = XMLResource(tree)
        validator = StreamingValidator(self.schema)
        self.assertTrue(validator.is_valid(resource))

    # -- memory guard ------------------------------------------------------

    def test_memory_limit_aborts(self):
        xml = '<root>' + '<item id="a" n="1">x</item>' * 50 + '</root>'
        # A deliberately tiny threshold and interval of one element.
        validator = StreamingValidator(
            self.schema, max_memory_bytes=5 * 1024 * 1024,
            memory_check_interval=1
        )
        with self.assertRaises(MemoryLimitError):
            list(validator.iter_errors(io.StringIO(xml)))

    # -- streaming / release behaviour ------------------------------------

    def test_elements_are_released(self):
        # The resource bookkeeping must not retain one entry per parsed node.
        xml = '<root>' + '<item id="i%d" n="1">x</item>' % 0
        xml += ''.join('<item id="i%d" n="1">x</item>' % i for i in range(1, 2000))
        xml += '</root>'
        validator = StreamingValidator(self.schema)
        resource = validator._get_resource(io.StringIO(xml))
        list(validator.iter_errors(resource))
        # Only the root namespace map survives; per-node entries are pruned.
        self.assertLessEqual(len(resource._nsmaps), 2)
        self.assertEqual(validator.element_count, 2001)
        # The child-resolution cache is bounded by distinct schema locations,
        # not by the number of rows.
        self.assertLessEqual(len(validator._child_cache), 1)

    def test_large_document_bounded_state(self):
        rows = 5000
        parts = ['<root>']
        for i in range(rows):
            parts.append('<item id="id%d" n="%d">x</item>' % (i, i))
        parts.append('</root>')
        validator = StreamingValidator(self.schema, memory_check_interval=512)
        errors = list(validator.iter_errors(io.StringIO(''.join(parts))))
        self.assertEqual([], errors)
        self.assertEqual(validator.element_count, rows + 1)
        # Cross-node state grows with the number of IDs, but the element
        # resolution cache stays constant for a uniform record layout.
        self.assertEqual(len(validator.ids), rows)
        self.assertEqual(len(validator._child_cache), 1)

    def test_target_namespace_document(self):
        schema = xmlschema.XMLSchema(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="http://example.test/ns"
                       elementFormDefault="qualified">
              <xs:element name="root">
                <xs:complexType><xs:sequence>
                  <xs:element name="item" maxOccurs="unbounded">
                    <xs:complexType><xs:simpleContent>
                      <xs:extension base="xs:string">
                        <xs:attribute name="id" type="xs:ID" use="required"/>
                      </xs:extension>
                    </xs:simpleContent></xs:complexType>
                  </xs:element>
                </xs:sequence></xs:complexType>
              </xs:element>
            </xs:schema>"""))
        good = ('<root xmlns="http://example.test/ns">'
                '<item id="a">x</item><item id="b">y</item></root>')
        dup = ('<root xmlns="http://example.test/ns">'
               '<item id="a"/><item id="a"/></root>')
        self.assertTrue(StreamingValidator(schema).is_valid(io.StringIO(good)))
        errors = list(StreamingValidator(schema).iter_errors(io.StringIO(dup)))
        self.assertEqual(1, len(errors))
        self.assertIn("duplicated xs:ID value 'a'", errors[0].reason)

    def test_choice_group_and_wildcard(self):
        schema = xmlschema.XMLSchema(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="root">
                <xs:complexType><xs:sequence>
                  <xs:choice maxOccurs="unbounded">
                    <xs:element name="name" type="xs:string"/>
                    <xs:element name="age" type="xs:int"/>
                    <xs:group ref="grp"/>
                    <xs:any namespace="##other" processContents="skip"/>
                  </xs:choice>
                </xs:sequence></xs:complexType>
              </xs:element>
              <xs:group name="grp">
                <xs:sequence><xs:element name="tag" type="xs:NCName"/></xs:sequence>
              </xs:group>
            </xs:schema>"""))
        good = ('<root><name>bob</name><age>30</age><tag>x1</tag>'
                '<ext:foo xmlns:ext="http://x.test">v</ext:foo></root>')
        self.assertTrue(StreamingValidator(schema).is_valid(io.StringIO(good)))
        bad = '<root><age>not-an-int</age></root>'
        errors = list(StreamingValidator(schema).iter_errors(io.StringIO(bad)))
        self.assertEqual(1, len(errors))
        self.assertIn('int', errors[0].reason)

    def test_depth_limit_enforced(self):
        from xmlschema import limits
        from xmlschema.exceptions import XMLResourceExceeded
        schema = xmlschema.XMLSchema(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="a"><xs:complexType><xs:sequence>
                <xs:element name="a" minOccurs="0"/>
              </xs:sequence></xs:complexType></xs:element>
            </xs:schema>"""))
        original = limits.MAX_XML_DEPTH
        limits.MAX_XML_DEPTH = 50
        try:
            with self.assertRaises(XMLResourceExceeded):
                StreamingValidator(schema).is_valid(
                    io.StringIO('<a>' * 200 + '</a>' * 200)
                )
        finally:
            limits.MAX_XML_DEPTH = original

    def test_deep_document_resolved_incrementally(self):
        schema = xmlschema.XMLSchema(dedent("""\
            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="a"><xs:complexType><xs:sequence>
                <xs:element name="a" minOccurs="0"><xs:complexType><xs:sequence>
                  <xs:element name="v" type="xs:int" minOccurs="0"/>
                </xs:sequence></xs:complexType></xs:element>
              </xs:sequence></xs:complexType></xs:element>
            </xs:schema>"""))
        depth = 200
        xml = '<a>' * depth + '<v>42</v>' + '</a>' * depth
        validator = StreamingValidator(schema)
        self.assertEqual([], list(validator.iter_errors(io.StringIO(xml))))
        self.assertEqual(validator.max_depth, depth + 1)
        # The cache holds only distinct schema locations, never one entry per node.
        self.assertLess(len(validator._child_cache), 5)

    def test_validator_reuse_resets_state(self):
        good = '<root><item id="a" n="1">x</item></root>'
        validator = StreamingValidator(self.schema)
        self.assertTrue(validator.is_valid(io.StringIO(good)))
        self.assertEqual({'a'}, validator.ids)
        # A second run must not keep the IDs collected on the first document.
        second = '<root><item id="z" n="2">x</item></root>'
        self.assertTrue(validator.is_valid(io.StringIO(second)))
        self.assertEqual({'z'}, validator.ids)
        # Counters reflect only the second document (root + one item).
        self.assertEqual(2, validator.element_count)
        self.assertEqual(2, validator.max_depth)


if __name__ == '__main__':
    header_template = 'Test streaming validator module'
    xmlschema.testing.run_xmlschema_tests(header_template)
